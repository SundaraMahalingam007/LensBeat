"""
signal/bpm.py — Peak detection, harmonic validation, BPM extraction (Phase 2 / Risk 3)

Full chain per the approach doc:
  Raw rPPG -> filtering.preprocess_waveform -> spectrum.compute_psd
    -> peak detection -> peak prominence/SNR -> harmonic checking -> BPM

Returns a BPMEstimate dataclass carrying not just the BPM value but the
diagnostics (SNR, prominence) that quality/signal_quality.py combines into
a confidence score -- so BPM extraction and confidence estimation share
one spectral analysis pass rather than recomputing it twice.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from signal_processing.filtering import preprocess_waveform
from signal_processing.spectrum import compute_psd, band_power


@dataclass
class BPMEstimate:
    bpm: float
    valid: bool                # False if no plausible peak was found at all
    peak_power: float
    total_band_power: float
    snr: float                 # peak_power / (total_band_power - peak_power)
    prominence: float          # normalized peak prominence, 0-1
    harmonic_corrected: bool   # True if a harmonic-doubling correction was applied


def _find_dominant_peak(freqs, psd, low_hz, high_hz):
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return None

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    peaks, props = find_peaks(band_psd, prominence=0.0)
    if len(peaks) == 0:
        # no local maxima -- fall back to the raw argmax
        idx = int(np.argmax(band_psd))
        return band_freqs[idx], band_psd[idx], 0.0

    prominences = props["prominences"]
    best = int(np.argmax(prominences))
    idx = peaks[best]
    return band_freqs[idx], band_psd[idx], float(prominences[best])


def _check_harmonic(freqs, psd, candidate_freq, candidate_power, low_hz):
    """If a lower-frequency peak exists at roughly half the candidate's
    frequency with comparable power, the candidate is likely the second
    harmonic of a slower true pulse -- prefer the fundamental instead.
    This directly targets Risk 3 (FFT selects the wrong peak / harmonics dominate)."""
    half_freq = candidate_freq / 2.0
    if half_freq < low_hz:
        return candidate_freq, candidate_power, False

    tolerance = 0.15  # Hz, allows for spectral resolution limits
    near_half = (freqs >= half_freq - tolerance) & (freqs <= half_freq + tolerance)
    if not np.any(near_half):
        return candidate_freq, candidate_power, False

    half_power = float(np.max(psd[near_half]))
    # if the sub-harmonic candidate carries at least 60% of the detected
    # peak's power, treat it as the true fundamental
    if half_power >= 0.6 * candidate_power:
        half_idx = np.argmax(psd[near_half])
        corrected_freq = freqs[near_half][half_idx]
        return corrected_freq, half_power, True

    return candidate_freq, candidate_power, False


def estimate_bpm(raw_waveform, fs=30, low_hz=0.83, high_hz=2.0, filter_low_hz=0.7, filter_high_hz=3.0):
    """Full Phase 2 pipeline. filter_low_hz/filter_high_hz are the wider
    bandpass-filtering range; low_hz/high_hz are the tighter physiological
    range used for peak search (matches the approach doc's 0.83-2.0 Hz /
    50-120 BPM default, distinct from the filter's wider passband)."""
    if len(raw_waveform) < 8:
        return BPMEstimate(0.0, False, 0.0, 0.0, 0.0, 0.0, False)

    filtered = preprocess_waveform(raw_waveform, fs=fs, low_hz=filter_low_hz, high_hz=filter_high_hz)
    freqs, psd = compute_psd(filtered, fs=fs)

    peak = _find_dominant_peak(freqs, psd, low_hz, high_hz)
    if peak is None:
        return BPMEstimate(0.0, False, 0.0, 0.0, 0.0, 0.0, False)

    peak_freq, peak_power, raw_prominence = peak
    corrected_freq, corrected_power, harmonic_corrected = _check_harmonic(freqs, psd, peak_freq, peak_power, low_hz)

    # SNR calculation: integrate power around the fundamental peak (accounting
    # for Welch spectral leakage) plus the second harmonic (characteristic of rPPG),
    # compared against the surrounding noise in the passband.
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    signal_half_width = max(4 * freq_resolution, 0.16)
    signal_low = corrected_freq - signal_half_width
    signal_high = corrected_freq + signal_half_width
    fund_power = band_power(freqs, psd, signal_low, signal_high)

    harm_power = 0.0
    harm_freq = 2.0 * corrected_freq
    if harm_freq <= filter_high_hz:
        harm_power = band_power(freqs, psd, harm_freq - signal_half_width, harm_freq + signal_half_width)

    signal_power = fund_power + harm_power
    total_power = band_power(freqs, psd, low_hz, high_hz)
    if harm_power > 0.0 and harm_freq > high_hz:
        total_power += band_power(freqs, psd, high_hz, min(harm_freq + signal_half_width, filter_high_hz))

    noise_power = max(total_power - signal_power, 1e-9)
    snr = signal_power / noise_power

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    max_psd_in_band = float(np.max(psd[band_mask])) if np.any(band_mask) else 1.0
    normalized_prominence = float(np.clip(raw_prominence / (max_psd_in_band + 1e-9), 0.0, 1.0))

    return BPMEstimate(
        bpm=float(corrected_freq * 60.0),
        valid=True,
        peak_power=corrected_power,
        total_band_power=total_power,
        snr=float(snr),
        prominence=normalized_prominence,
        harmonic_corrected=harmonic_corrected,
    )
