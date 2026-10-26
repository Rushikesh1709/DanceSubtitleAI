"""Pose estimation with MediaPipe Pose (pretrained, bundled model)."""
from typing import Optional

import cv2
import numpy as np


class PoseBackendError(RuntimeError):
    """Raised when MediaPipe is missing or incompatible (message is user-safe)."""


class LM:
    """MediaPipe Pose landmark indices used by this project (left/right = the dancer's own sides)."""
    NOSE = 0
    L_EAR, R_EAR = 7, 8
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_PINKY, R_PINKY = 17, 18
    L_INDEX, R_INDEX = 19, 20
    L_THUMB, R_THUMB = 21, 22
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28


NUM_POSE_LANDMARKS = 33

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32), (15, 19), (16, 20), (0, 11), (0, 12),
]


def load_mediapipe_solutions():
    """Import MediaPipe's classic `solutions` API or raise a clear PoseBackendError."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise PoseBackendError(
            "MediaPipe is not installed. Run: python -m pip install -r requirements.txt") from exc
    if not hasattr(mp, "solutions"):
        raise PoseBackendError(
            "The installed MediaPipe version does not include the classic 'solutions' API. "
            "Install the tested version with: python -m pip install mediapipe==0.10.14")
    return mp.solutions


class PoseExtractor:
    """Runs MediaPipe Pose on BGR frames and returns a (33, 4) array [x, y, z, visibility]."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, max_dim: int = 960):
        solutions = load_mediapipe_solutions()
        self.max_dim = max_dim
        self._pose = solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_dim / float(max(h, w)))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        pts = result.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.z, p.visibility] for p in pts], dtype=np.float32)

    def close(self):
        try:
            self._pose.close()
        except Exception:
            pass
