"""
benchmark.py — run the pipeline against labeled test clips and report metrics.

Usage:
    python benchmark.py --version v1_baseline --clips-dir /path/to/test_clips \\
        --model-path LensBeat_PhysNet_finetuned.pth

Expects `clips-dir` to contain one subfolder per test clip, each with:
    frames.npy   -- (T, H, W, 3) uint8 RGB frames (pre-extracted, not raw video)
    true_bpm.txt -- single float, ground-truth BPM for this clip

This intentionally works from pre-extracted frame arrays rather than raw
video files, following the lesson from LensBeat's original development:
video decoding (OpenCV specifically) was a recurring source of crashes, so
evaluation tooling here sidesteps that entirely.

This implements Section 4 ("create a baseline dataset... measure MAE, RMSE,
Pearson, ±5/±10 BPM") and Section 6 (log every inference for later analysis).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from inference.pipeline import LensBeatPipeline
from evaluation.metrics import compute_all_metrics
from evaluation.benchmark import print_dashboard, log_version


def load_clip(clip_dir):
    frames = np.load(clip_dir / "frames.npy")  # (T, H, W, 3) uint8 RGB
    with open(clip_dir / "true_bpm.txt") as f:
        true_bpm = float(f.read().strip())
    return frames, true_bpm


def run_benchmark(clips_dir, model_path, roi_mode="landmark_multi", chunk_length=128,
                   frame_size=72, fs=30, log_path="evaluation/version_log.jsonl", version_name="unnamed"):
    pipeline = LensBeatPipeline(
        model_path=model_path,
        frame_size=frame_size,
        chunk_length=chunk_length,
        fs=fs,
        roi_mode=roi_mode,
        infer_stride=1,  # evaluate every available chunk, not throttled like live deployment
    )

    pred_bpms, true_bpms = [], []
    log_rows = []

    clip_dirs = sorted([p for p in Path(clips_dir).iterdir() if p.is_dir()])
    print(f"Found {len(clip_dirs)} test clips.")

    for clip_dir in clip_dirs:
        try:
            frames, true_bpm = load_clip(clip_dir)
        except Exception as e:
            print(f"  SKIP {clip_dir.name}: {e}")
            continue

        pipeline.frame_buffer = []
        pipeline.motion_scores = []
        pipeline._frames_since_infer = pipeline.infer_stride  # force inference on first full buffer

        result = None
        for frame_rgb in frames[:chunk_length]:
            frame_bgr = frame_rgb[:, :, ::-1]  # pipeline expects BGR (matches cv2 webcam frames)
            result = pipeline.process_frame(frame_bgr)

        if result is None or not result["valid"]:
            print(f"  {clip_dir.name}: no valid BPM estimate, skipping")
            continue

        pred_bpms.append(result["raw_bpm"])
        true_bpms.append(true_bpm)
        log_rows.append({
            "clip": clip_dir.name,
            "true_bpm": true_bpm,
            "pred_bpm": result["raw_bpm"],
            "signal_quality": result["signal_quality"],
            "confidence": result["confidence"],
            "motion_level": result["motion_level"],
        })
        print(f"  {clip_dir.name}: pred={result['raw_bpm']:.1f}  true={true_bpm:.1f}  "
              f"quality={result['signal_quality']}")

    pipeline.close()

    if not pred_bpms:
        print("No valid predictions produced -- check clip format and model path.")
        return

    metrics = compute_all_metrics(pred_bpms, true_bpms)
    print()
    print_dashboard(metrics, title=f"LensBeat Validation — {version_name}")

    log_version(
        log_path=log_path,
        version_name=version_name,
        dataset=str(clips_dir),
        learning_rate=None,
        epochs=None,
        roi_mode=roi_mode,
        augmentation=None,
        metrics=metrics,
    )
    print(f"\nLogged to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="unnamed", help="Version label for the log (e.g. v1_baseline)")
    parser.add_argument("--clips-dir", required=True, help="Directory of test clip subfolders")
    parser.add_argument("--model-path", required=True, help="Path to the PhysNet .pth checkpoint")
    parser.add_argument("--roi-mode", default="landmark_multi",
                         choices=["face_crop", "padded_crop_10", "padded_crop_20", "landmark_multi"])
    args = parser.parse_args()

    run_benchmark(
        clips_dir=args.clips_dir,
        model_path=args.model_path,
        roi_mode=args.roi_mode,
        version_name=args.version,
    )
