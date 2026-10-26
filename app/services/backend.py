"""Combines pose, hand and face detectors into one per-frame analyser."""
from typing import Dict, Optional

import numpy as np

from .pose_service import PoseExtractor


class MediaPipeBackend:
    """Default backend. Tests can replace it with any object that has `analyze` and `close`."""

    def __init__(self, enable_hands: bool = True, enable_face: bool = False, model_complexity: int = 1):
        self.pose = PoseExtractor(model_complexity=model_complexity)
        self.hands = None
        self.face = None
        if enable_hands:
            from .hand_gesture_service import HandShapeDetector
            self.hands = HandShapeDetector()
        if enable_face:
            from .expression_service import FaceExpressionDetector
            self.face = FaceExpressionDetector()

    def analyze(self, frame_bgr: np.ndarray) -> Dict[str, Optional[object]]:
        pose = self.pose.process(frame_bgr)
        hands = {"left": None, "right": None}
        face = None
        if pose is not None:
            if self.hands is not None:
                hands = self.hands.detect(frame_bgr, pose)
            if self.face is not None:
                face = self.face.detect(frame_bgr, pose)
        return {"pose": pose, "hands": hands, "face": face}

    def close(self):
        for part in (self.pose, self.hands, self.face):
            if part is not None:
                part.close()


def create_default_backend(options: dict) -> MediaPipeBackend:
    return MediaPipeBackend(enable_hands=bool(options.get("enable_hands")),
                            enable_face=bool(options.get("enable_face")),
                            model_complexity=int(options.get("model_complexity", 1)))
