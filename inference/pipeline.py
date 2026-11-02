"""
inference/pipeline.py — Orchestrates the full measurement pipeline

Implements the "Final Architecture" from the approach doc (Section 11):

  RGB Webcam -> Face Detection/Tracking -> ROI Extraction -> Preprocessing
    -> PhysNet -> rPPG waveform -> Signal Processing -> BPM Estimation
    -> Temporal Stabilization -> Quality/Confidence -> Output

This module has no Flask/SocketIO dependency -- app.py wires it to the web
layer. Keeping the pipeline separate from the transport is exactly the
Section 7 "replace one component without rewriting the application" goal:
this same class works in a Flask app, a batch-evaluation script, or a CLI tool.
"""

import numpy as np
import torch

from model.physnet import PhysNet_padding_Encoder_Decoder_MAX
from preprocessing.roi import ROIExtractor
from signal_processing.bpm import estimate_bpm
from signal_processing.stabilization import TemporalStabilizer
from quality.signal_quality import assess_quality


class LensBeatPipeline:
    def __init__(self, model_path, frame_size=72, chunk_length=128, fs=30,
                 bpm_low_hz=0.83, bpm_high_hz=2.0, filter_low_hz=0.7, filter_high_hz=3.0,
                 roi_mode="landmark_multi", infer_stride=32, device=None):
        self.frame_size = frame_size
        self.chunk_length = chunk_length
        self.fs = fs
        self.bpm_low_hz = bpm_low_hz
        self.bpm_high_hz = bpm_high_hz
        self.filter_low_hz = filter_low_hz
        self.filter_high_hz = filter_high_hz
        self.infer_stride = infer_stride
        self._frames_since_infer = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=chunk_length).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.roi_extractor = ROIExtractor(frame_size=frame_size, mode=roi_mode)
        self.stabilizer = TemporalStabilizer()

        self.frame_buffer = []
        self.motion_scores = []  # parallels frame_buffer, one score per frame

    def process_frame(self, frame_bgr):
        """Feed one new webcam frame. Returns a result dict once the buffer
        is full AND infer_stride frames have passed since the last inference
        run; otherwise returns None (buffering, or throttled by stride)."""
        roi, motion, detected = self.roi_extractor.extract(frame_bgr)
        roi_rgb = roi[:, :, ::-1].astype(np.float32) / 255.0

        self.frame_buffer.append(roi_rgb)
        self.motion_scores.append(motion)
        if len(self.frame_buffer) > self.chunk_length:
            self.frame_buffer.pop(0)
            self.motion_scores.pop(0)

        if len(self.frame_buffer) < self.chunk_length:
            return None

        self._frames_since_infer += 1
        if self._frames_since_infer < self.infer_stride:
            return None
        self._frames_since_infer = 0

        return self._run_inference(face_detected=detected)

    @torch.no_grad()
    def _run_inference(self, face_detected):
        clip = np.stack(self.frame_buffer)
        clip = (clip - clip.mean()) / (clip.std() + 1e-6)
        clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2).float().unsqueeze(0).to(self.device)

        pred = self.model(clip_t)
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred = pred.squeeze() if pred.dim() > 1 else pred
        waveform = pred.cpu().numpy()

        bpm_estimate = estimate_bpm(
            waveform, fs=self.fs,
            low_hz=self.bpm_low_hz, high_hz=self.bpm_high_hz,
            filter_low_hz=self.filter_low_hz, filter_high_hz=self.filter_high_hz,
        )

        stabilized_bpm = self.stabilizer.update(bpm_estimate.bpm, valid=bpm_estimate.valid)
        avg_motion = float(np.mean(self.motion_scores)) if self.motion_scores else 0.0

        quality = assess_quality(
            snr=bpm_estimate.snr,
            prominence=bpm_estimate.prominence,
            temporal_std_bpm=self.stabilizer.recent_std(),
            motion_score=avg_motion,
            face_detected=face_detected,
        )

        return {
            "bpm": round(stabilized_bpm, 1),
            "raw_bpm": round(bpm_estimate.bpm, 1),
            "valid": bpm_estimate.valid,
            "harmonic_corrected": bpm_estimate.harmonic_corrected,
            "face_detected": face_detected,
            "confidence": quality.confidence,
            "signal_quality": quality.label,
            "motion_level": "HIGH" if avg_motion > 0.02 else ("MODERATE" if avg_motion > 0.008 else "LOW"),
            "waveform": waveform.tolist(),
        }

    def close(self):
        self.roi_extractor.close()
