"""
preprocessing/roi.py — ROI extraction strategies (approach doc Phase 4)

Implements the versions the approach doc specifies for comparison:
  Version A: plain face bounding-box crop
  Version B/C: face crop + padding (10% / 20%)
  Version D: landmark-based multi-region (forehead + left cheek + right cheek),
             combined with configurable weights

Version D is the default / target architecture (Section 11), since it isolates
higher-PPG-yield skin regions instead of treating the whole face box as
uniformly useful (Risk 1). All versions fall back to a center-crop if no
face has ever been detected, so the pipeline degrades gracefully rather
than hard-failing.
"""

import cv2
import numpy as np

from preprocessing.face import FaceTracker


def _center_crop_resize(frame_bgr, frame_size):
    h, w, _ = frame_bgr.shape
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (frame_size, frame_size))


def _bbox_from_landmarks(landmarks_px, indices, frame_shape, pad_frac=0.0):
    h, w, _ = frame_shape
    pts = landmarks_px[indices]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad_frac
    x1 += bw * pad_frac
    y0 -= bh * pad_frac
    y1 += bh * pad_frac
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, x1, y1


class ROIExtractor:
    """
    mode:
      "center_crop"       -- matches original training preprocessing exactly
                             (blind center-crop, no face detection) -- use this
                             with a checkpoint trained that way until a
                             retrained/fine-tuned checkpoint exists for the
                             other modes
      "face_crop"         -- Version A: plain face bounding box
      "padded_crop_10"    -- Version B: face box + 10% padding
      "padded_crop_20"    -- Version C: face box + 20% padding
      "landmark_multi"    -- Version D: forehead + cheeks, weighted combination
    """

    VALID_MODES = ("center_crop", "face_crop", "padded_crop_10", "padded_crop_20", "landmark_multi")

    def __init__(self, frame_size=72, mode="landmark_multi",
                 region_weights=(0.5, 0.25, 0.25)):  # (forehead, left cheek, right cheek)
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.frame_size = frame_size
        self.mode = mode
        self.region_weights = region_weights
        self.tracker = FaceTracker()
        self._last_landmarks = None

    def _face_bbox(self, landmarks_px, frame_shape, pad_frac):
        return _bbox_from_landmarks(landmarks_px, FaceTracker.FACE_OVAL_LANDMARKS, frame_shape, pad_frac)

    def extract(self, frame_bgr):
        """Returns (roi_bgr_resized, motion_score, detected: bool)."""
        h, w, _ = frame_bgr.shape

        landmarks_px, detected = self.tracker.process(frame_bgr)

        motion = 0.0
        if landmarks_px is not None:
            frame_diag = float(np.hypot(h, w))
            motion = self.tracker.motion_score(self._last_landmarks, landmarks_px, frame_diag)
            self._last_landmarks = landmarks_px

        if self.mode == "center_crop":
            return _center_crop_resize(frame_bgr, self.frame_size), motion, detected

        if landmarks_px is None:
            return _center_crop_resize(frame_bgr, self.frame_size), motion, False

        if self.mode == "face_crop":
            x0, y0, x1, y1 = self._face_bbox(landmarks_px, frame_bgr.shape, pad_frac=0.0)
        elif self.mode == "padded_crop_10":
            x0, y0, x1, y1 = self._face_bbox(landmarks_px, frame_bgr.shape, pad_frac=0.10)
        elif self.mode == "padded_crop_20":
            x0, y0, x1, y1 = self._face_bbox(landmarks_px, frame_bgr.shape, pad_frac=0.20)
        else:  # landmark_multi
            return self._extract_multi_region(frame_bgr, landmarks_px), motion, detected

        if x1 <= x0 or y1 <= y0:
            return _center_crop_resize(frame_bgr, self.frame_size), motion, detected
        crop = frame_bgr[y0:y1, x0:x1]
        return cv2.resize(crop, (self.frame_size, self.frame_size)), motion, detected

    def _extract_multi_region(self, frame_bgr, landmarks_px):
        """Version D: extract forehead + left cheek + right cheek separately,
        resize each to the same size, and combine with region_weights.
        Combining at the pixel level (weighted average) keeps a single
        (H, W, 3) frame flowing into PhysNet unchanged -- no architecture change needed."""
        regions = [
            FaceTracker.FOREHEAD_LANDMARKS,
            FaceTracker.LEFT_CHEEK_LANDMARKS,
            FaceTracker.RIGHT_CHEEK_LANDMARKS,
        ]
        crops = []
        for region_landmarks in regions:
            x0, y0, x1, y1 = _bbox_from_landmarks(landmarks_px, region_landmarks, frame_bgr.shape, pad_frac=0.15)
            if x1 <= x0 or y1 <= y0:
                crops.append(None)
                continue
            crop = frame_bgr[y0:y1, x0:x1]
            crops.append(cv2.resize(crop, (self.frame_size, self.frame_size)).astype(np.float32))

        valid = [(c, w) for c, w in zip(crops, self.region_weights) if c is not None]
        if not valid:
            return _center_crop_resize(frame_bgr, self.frame_size)

        total_w = sum(w for _, w in valid)
        combined = sum(c * (w / total_w) for c, w in valid)
        return combined.astype(np.uint8)

    def close(self):
        self.tracker.close()
