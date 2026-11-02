"""
signal/spectrum.py — Welch PSD spectral estimation (Phase 2)

Welch's method averages overlapping windowed FFTs, giving a smoother, less
noisy power spectrum than a single raw FFT -- particularly relevant for a
~4 second (128-frame @ 30fps) clip, where a plain FFT has poor frequency
resolution and is more susceptible to a single noisy segment dominating
the result.
"""

import numpy as np
from scipy.signal import welch


def compute_psd(waveform, fs=30, nperseg=None, nfft=1024):
    """Returns (freqs, psd). nperseg defaults to min(len(waveform), fs*2)
    -- a 2-second sub-window, capped by the signal's actual length.

    nfft zero-pads each segment to a much finer frequency grid than nperseg
    alone would give (e.g. nperseg=60 at fs=30 only yields 0.5 Hz bins --
    coarse enough to blur 60 vs 75 vs 90 BPM together). This doesn't add
    real spectral information, but it lets the peak-finding in signal_processing/bpm.py
    localize the true peak far more precisely instead of being stuck on a
    handful of widely-spaced bins."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if nperseg is None:
        nperseg = min(len(waveform), fs * 2)
    nperseg = max(nperseg, 8)  # scipy.welch needs a minimum segment length
    nfft = max(nfft, nperseg)
    freqs, psd = welch(waveform, fs=fs, nperseg=nperseg, nfft=nfft)
    return freqs, psd


def band_power(freqs, psd, low_hz, high_hz):
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return 0.0
    # np.trapz was removed in NumPy 2.0+ (renamed to np.trapezoid); support both
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(psd[band], freqs[band]))
