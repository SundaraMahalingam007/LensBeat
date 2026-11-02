"""
preprocessing/face.py — Face detection + tracking (mediapipe Face Landmarker / Mesh)

Per the approach doc's Section 7 target ("Haar -> MediaPipe" swap) and
Risk 2 mitigation (face tracking). Supports modern MediaPipe Tasks (Python 3.12+),
legacy solutions.face_mesh (Python <= 3.11), and OpenCV Haar Cascade fallback.

FaceTracker wraps face landmarker and adds EMA smoothing over landmark
positions across frames, so the ROI doesn't jitter frame-to-frame even
though landmark detection re-detects independently on every frame.
"""

import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp


class FaceTracker:
    """Detects face landmarks per frame and smooths their positions over time.

    Returns None on a frame with no detected face, letting the caller decide
    how to handle a miss (e.g. reuse last known ROI, flag low signal quality).
    """

    # A small, stable subset of mediapipe FaceMesh's 468 landmarks used to
    # anchor the forehead and cheek regions. Indices per the standard
    # mediapipe face mesh topology.
    FOREHEAD_LANDMARKS = [10, 108, 151, 337]      # upper-forehead band
    LEFT_CHEEK_LANDMARKS = [50, 101, 118, 187]     # subject's left cheek
    RIGHT_CHEEK_LANDMARKS = [280, 330, 347, 411]   # subject's right cheek
    FACE_OVAL_LANDMARKS = [10, 152, 234, 454]      # top, chin, left, right extremes

    def __init__(self, smoothing_alpha=0.4, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.smoothing_alpha = smoothing_alpha
        self._smoothed_landmarks = None  # (N, 2) pixel coords, EMA-smoothed
        self._frames_since_detect = 0
        self.MAX_FRAMES_WITHOUT_DETECTION = 10  # give up reusing stale landmarks after this many misses

        self._backend = None
        self._mesh = None
        self._detector = None
        self._cascade = None

        # 1. Try legacy mediapipe.solutions (Python <= 3.11)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._backend = "solutions"
            except Exception:
                self._mesh = None

        # 2. Try modern mediapipe.tasks (Python 3.12+)
        if self._backend is None:
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                task_path = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
                task_path = os.path.abspath(task_path)
                if not os.path.exists(task_path):
                    if os.path.exists("face_landmarker.task"):
                        task_path = os.path.abspath("face_landmarker.task")
                    else:
                        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                        urllib.request.urlretrieve(url, task_path)

                base_options = mp_python.BaseOptions(model_asset_path=task_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._detector = mp_vision.FaceLandmarker.create_from_options(options)
                self._backend = "tasks"
            except Exception:
                self._detector = None

        # 3. Fallback to OpenCV Haar cascade
        if self._backend is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            self._backend = "cascade"

    def process(self, frame_bgr):
        """Returns (landmarks_px, detected: bool).
        landmarks_px is an (468, 2) array of smoothed pixel coordinates, or
        None if no face has ever been detected (or too many consecutive misses)."""
        h, w, _ = frame_bgr.shape
        raw = None

        if self._backend == "solutions" and self._mesh is not None:
            rgb = frame_bgr[:, :, ::-1]
            results = self._mesh.process(rgb)
            if results.multi_face_landmarks:
                raw = np.array(
                    [[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark],
                    dtype=np.float32,
                )
        elif self._backend == "tasks" and self._detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._detector.detect(mp_image)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                raw = np.array(
                    [[lm.x * w, lm.y * h] for lm in results.face_landmarks[0]],
                    dtype=np.float32,
                )
        elif self._backend == "cascade" and self._cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                synth = np.zeros((468, 2), dtype=np.float32)
                synth[10] = [fx + fw * 0.5, fy]
                synth[152] = [fx + fw * 0.5, fy + fh]
                synth[234] = [fx, fy + fh * 0.5]
                synth[454] = [fx + fw, fy + fh * 0.5]
                synth[108] = [fx + fw * 0.3, fy + fh * 0.15]
                synth[151] = [fx + fw * 0.5, fy + fh * 0.15]
                synth[337] = [fx + fw * 0.7, fy + fh * 0.15]
                for idx in self.LEFT_CHEEK_LANDMARKS:
                    synth[idx] = [fx + fw * 0.25, fy + fh * 0.55]
                for idx in self.RIGHT_CHEEK_LANDMARKS:
                    synth[idx] = [fx + fw * 0.75, fy + fh * 0.55]
                raw = synth

        if raw is not None:
            if self._smoothed_landmarks is None:
                self._smoothed_landmarks = raw
            else:
                a = self.smoothing_alpha
                self._smoothed_landmarks = a * raw + (1 - a) * self._smoothed_landmarks
            self._frames_since_detect = 0
            return self._smoothed_landmarks, True

        self._frames_since_detect += 1
        if self._smoothed_landmarks is not None and self._frames_since_detect <= self.MAX_FRAMES_WITHOUT_DETECTION:
            return self._smoothed_landmarks, False  # reuse last known position, flagged as a miss
        return None, False

    def motion_score(self, prev_landmarks, curr_landmarks, frame_diag):
        """Mean landmark displacement, normalized by frame diagonal so the
        score is roughly resolution-independent. Used by quality/signal_quality.py."""
        if prev_landmarks is None or curr_landmarks is None:
            return 0.0
        disp = np.linalg.norm(curr_landmarks - prev_landmarks, axis=1)
        return float(np.mean(disp) / (frame_diag + 1e-6))

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
        if self._detector is not None:
            self._detector.close()
