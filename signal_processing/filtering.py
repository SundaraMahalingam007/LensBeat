"""
signal/filtering.py — Phase 2: rPPG -> Mean removal -> Detrend -> Butterworth bandpass
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend as scipy_detrend


def remove_mean(waveform):
    return waveform - np.mean(waveform)


def detrend(waveform):
    """Removes linear drift -- PhysNet's raw output can carry slow baseline
    wander that isn't part of the pulse signal."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if len(waveform) < 4:
        return waveform
    return scipy_detrend(waveform, type="linear")


def bandpass_filter(waveform, fs=30, low_hz=0.7, high_hz=3.0, order=3):
    """Butterworth bandpass restricted to the physiologically plausible
    heart-rate frequency range, removing noise and motion artifacts outside it."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if len(waveform) < order * 3:
        return waveform  # too short to filter meaningfully -- return as-is
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, waveform)


def preprocess_waveform(waveform, fs=30, low_hz=0.7, high_hz=3.0):
    """Full Phase 2 chain: mean removal -> detrend -> bandpass."""
    w = remove_mean(waveform)
    w = detrend(w)
    w = bandpass_filter(w, fs=fs, low_hz=low_hz, high_hz=high_hz)
    return w
