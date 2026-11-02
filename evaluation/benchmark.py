"""
evaluation/benchmark.py — Section 5 dashboard + Section 8 model versioning

Prints the exact dashboard format the approach doc specifies, and keeps a
simple append-only log of every benchmarked version so "we changed
something and don't know which version was better" (Section 8) can't happen.
"""

import json
import os
from datetime import datetime

from evaluation.metrics import compute_all_metrics


def print_dashboard(metrics, signal_quality_label=None, motion_level=None, title="LensBeat Validation"):
    print(title)
    print("-" * len(title))
    print(f"MAE             : {metrics['mae']:.1f} BPM")
    print(f"RMSE            : {metrics['rmse']:.1f} BPM")
    print(f"Pearson r       : {metrics['pearson_r']:.2f}")
    print(f"Within ±5 BPM   : {metrics['within_5_bpm']:.0f}%")
    print(f"Within ±10 BPM  : {metrics['within_10_bpm']:.0f}%")
    if signal_quality_label:
        print()
        print(f"Signal Quality  : {signal_quality_label}")
    if motion_level:
        print(f"Motion Level    : {motion_level}")


def log_version(log_path, version_name, dataset, learning_rate, epochs, roi_mode,
                 augmentation, metrics):
    """Appends one line of JSON per version to a log file -- Section 8's
    "maintain: version, dataset, learning rate, epochs, ROI, augmentation,
    MAE, RMSE, ±5 BPM" requirement."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": version_name,
        "dataset": dataset,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "roi_mode": roi_mode,
        "augmentation": augmentation,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "pearson_r": metrics["pearson_r"],
        "within_5_bpm": metrics["within_5_bpm"],
        "within_10_bpm": metrics["within_10_bpm"],
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_version_log(log_path):
    """Reads back every logged version for comparison (e.g. building the
    'Version | MAE | RMSE | ±5 BPM | ±10 BPM' table from Section 4's baseline)."""
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_version_table(log_path):
    entries = load_version_log(log_path)
    if not entries:
        print("No versions logged yet.")
        return
    print(f"{'Version':<20}{'MAE':>8}{'RMSE':>8}{'±5 BPM':>10}{'±10 BPM':>10}")
    for e in entries:
        print(f"{e['version']:<20}{e['mae']:>8.1f}{e['rmse']:>8.1f}"
              f"{e['within_5_bpm']:>9.0f}%{e['within_10_bpm']:>9.0f}%")
