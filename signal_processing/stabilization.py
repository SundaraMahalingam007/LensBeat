"""
signal/stabilization.py — Phase 3: temporal stabilization (Risk 4)

Chain per approach doc:
  New BPM -> validity check -> median of recent estimates -> EMA -> displayed BPM
"""

from collections import deque


class TemporalStabilizer:
    def __init__(self, median_window=5, ema_alpha=0.25, min_bpm=40.0, max_bpm=180.0):
        self.history = deque(maxlen=median_window)
        self.ema_alpha = ema_alpha
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self._ema_value = None

    def _validity_check(self, bpm):
        return self.min_bpm <= bpm <= self.max_bpm

    def update(self, raw_bpm, valid=True):
        """Returns the stabilized (displayed) BPM. `valid` lets the caller
        pass through the BPMEstimate.valid flag from signal_processing/bpm.py so a
        failed peak-detection doesn't corrupt the history."""
        if valid and self._validity_check(raw_bpm):
            self.history.append(raw_bpm)

        if not self.history:
            return 0.0

        median_bpm = sorted(self.history)[len(self.history) // 2]

        if self._ema_value is None:
            self._ema_value = median_bpm
        else:
            self._ema_value = self.ema_alpha * median_bpm + (1 - self.ema_alpha) * self._ema_value

        return float(self._ema_value)

    def current(self):
        return float(self._ema_value) if self._ema_value is not None else 0.0

    def recent_std(self):
        """Standard deviation of recent accepted estimates -- used by
        quality/signal_quality.py as a temporal-consistency signal."""
        if len(self.history) < 2:
            return 0.0
        vals = list(self.history)
        mean = sum(vals) / len(vals)
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
