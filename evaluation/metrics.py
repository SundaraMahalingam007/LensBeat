"""
evaluation/metrics.py — Section 5: MAE, RMSE, Pearson correlation, ±5/±10 BPM
"""

import numpy as np
from scipy.stats import pearsonr


def mae(pred_bpms, true_bpms):
    pred_bpms, true_bpms = np.asarray(pred_bpms), np.asarray(true_bpms)
    return float(np.mean(np.abs(pred_bpms - true_bpms)))


def rmse(pred_bpms, true_bpms):
    pred_bpms, true_bpms = np.asarray(pred_bpms), np.asarray(true_bpms)
    return float(np.sqrt(np.mean((pred_bpms - true_bpms) ** 2)))


def pearson_correlation(pred_bpms, true_bpms):
    pred_bpms, true_bpms = np.asarray(pred_bpms), np.asarray(true_bpms)
    if len(pred_bpms) < 2:
        return 0.0
    corr, _ = pearsonr(pred_bpms, true_bpms)
    return float(corr)


def within_tolerance(pred_bpms, true_bpms, tolerance_bpm):
    pred_bpms, true_bpms = np.asarray(pred_bpms), np.asarray(true_bpms)
    return float(np.mean(np.abs(pred_bpms - true_bpms) <= tolerance_bpm) * 100)


def compute_all_metrics(pred_bpms, true_bpms):
    """Returns a dict with every metric the approach doc's Section 5 dashboard needs."""
    return {
        "mae": mae(pred_bpms, true_bpms),
        "rmse": rmse(pred_bpms, true_bpms),
        "pearson_r": pearson_correlation(pred_bpms, true_bpms),
        "within_5_bpm": within_tolerance(pred_bpms, true_bpms, 5),
        "within_10_bpm": within_tolerance(pred_bpms, true_bpms, 10),
    }
