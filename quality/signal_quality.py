"""
quality/signal_quality.py — Phase 7: confidence and quality layer (Risk 5)

Combines the four indicators the approach doc names:
  - Spectral SNR            (from signal_processing/bpm.py's BPMEstimate)
  - Peak prominence         (from signal_processing/bpm.py's BPMEstimate)
  - Temporal consistency    (from signal_processing/stabilization.py's recent_std)
  - Motion score            (from preprocessing/roi.py's per-frame motion score)

into a single 0-100 confidence score and a GOOD/FAIR/LOW label, so the UI
never presents a bare BPM number as if it were unconditionally reliable.
"""

from dataclasses import dataclass


@dataclass
class QualityAssessment:
    confidence: float          # 0-100
    label: str                 # "GOOD" | "FAIR" | "LOW"
    snr: float
    prominence: float
    temporal_std: float
    motion_score: float


def _normalize_snr(snr, snr_good=0.70, snr_min=0.15):
    """Calibrated rPPG spectral SNR: snr_good (0.70+) maps to 1.0; noise floor (<=0.15) maps to 0.0."""
    if snr <= snr_min:
        return 0.0
    return min((snr - snr_min) / (snr_good - snr_min), 1.0)


def _normalize_prominence(prominence, prom_good=0.60, prom_min=0.15):
    """Calibrated spectral peak prominence within physiological band: 0.60+ is a sharp dominant peak."""
    if prominence <= prom_min:
        return 0.0
    return min((prominence - prom_min) / (prom_good - prom_min), 1.0)


def _normalize_temporal_std(std_bpm, std_good=3.5, std_bad=15.0):
    """Natural resting heart rate variability (<=3.5 BPM) is not penalized. Jitter above std_bad collapses."""
    if std_bpm <= std_good:
        return 1.0
    if std_bpm >= std_bad:
        return 0.0
    return 1.0 - (std_bpm - std_good) / (std_bad - std_good)


def _normalize_motion(motion_score, motion_still=0.006, motion_bad=0.025):
    """motion_score <= 0.006 indicates a still subject (1.0). Above 0.025 indicates active head movement."""
    if motion_score <= motion_still:
        return 1.0
    if motion_score >= motion_bad:
        return 0.0
    return 1.0 - (motion_score - motion_still) / (motion_bad - motion_still)


def assess_quality(snr, prominence, temporal_std_bpm, motion_score,
                    face_detected=True,
                    weights=(0.35, 0.25, 0.25, 0.15)):
    """weights order: (snr, prominence, temporal_consistency, motion)."""
    if not face_detected:
        return QualityAssessment(
            confidence=0.0,
            label="LOW",
            snr=snr,
            prominence=prominence,
            temporal_std=temporal_std_bpm,
            motion_score=motion_score,
        )

    w_snr, w_prom, w_temporal, w_motion = weights

    norm_snr = _normalize_snr(snr)
    norm_prom = _normalize_prominence(prominence)
    norm_temporal = _normalize_temporal_std(temporal_std_bpm)
    norm_motion = _normalize_motion(motion_score)

    score = (
        w_snr * norm_snr
        + w_prom * norm_prom
        + w_temporal * norm_temporal
        + w_motion * norm_motion
    )
    confidence = round(score * 100, 1)

    if confidence >= 65:
        label = "GOOD"
    elif confidence >= 40:
        label = "FAIR"
    else:
        label = "LOW"

    return QualityAssessment(
        confidence=confidence,
        label=label,
        snr=snr,
        prominence=prominence,
        temporal_std=temporal_std_bpm,
        motion_score=motion_score,
    )
