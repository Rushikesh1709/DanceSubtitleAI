"""Optional facial-expression ESTIMATES from MediaPipe Face Mesh geometry.

These are geometric heuristics measured relative to the dancer's own median face in the same
video. They describe visible facial movement (smile, raised eyebrows, open mouth); they do NOT
prove an emotion or a rasa.
"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .pose_service import LM, load_mediapipe_solutions
from .utils import group_runs, mode_filter, sample_edges


def _d(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def face_metrics(lm: np.ndarray) -> Optional[dict]:
    """Scale-free geometry from 468 face-mesh points (x, y in an isotropic frame)."""
    face_w, face_h = _d(lm[234], lm[454]), _d(lm[10], lm[152])
    if face_w < 1e-6 or face_h < 1e-6:
        return None
    nose_pos = (lm[1][0] - lm[234][0]) / (lm[454][0] - lm[234][0] + 1e-9)
    lip_y = (lm[13][1] + lm[14][1]) / 2.0
    corner_y = (lm[61][1] + lm[291][1]) / 2.0
    return {
        "frontal": abs(nose_pos - 0.5) < 0.18,
        "mouth_width": _d(lm[61], lm[291]) / face_w,
        "mouth_open": _d(lm[13], lm[14]) / face_h,
        "corner_lift": (lip_y - corner_y) / face_h,
        "brow_gap": (_d(lm[105], lm[159]) + _d(lm[334], lm[386])) / 2.0 / face_h,
    }


def classify_expressions(metrics: Sequence[Optional[dict]], min_baseline_samples: int = 8,
                         window: int = 5) -> List[Optional[str]]:
    """Label each sample 'smiling' / 'raised_eyebrows' / 'open_mouth' / 'neutral' / None.

    'neutral' means "close to this dancer's typical face in this video".
    """
    usable = [m for m in metrics if m and m.get("frontal")]
    if len(usable) < min_baseline_samples:
        return [None] * len(metrics)
    base = {k: float(np.median([m[k] for m in usable]))
            for k in ("mouth_width", "mouth_open", "corner_lift", "brow_gap")}
    labels: List[Optional[str]] = []
    for m in metrics:
        if not m or not m.get("frontal"):
            labels.append(None)
        elif m["corner_lift"] > base["corner_lift"] + 0.012 and m["mouth_width"] > base["mouth_width"] * 1.06:
            labels.append("smiling")
        elif m["brow_gap"] > base["brow_gap"] * 1.15:
            labels.append("raised_eyebrows")
        elif m["mouth_open"] > base["mouth_open"] + 0.05:
            labels.append("open_mouth")
        else:
            labels.append("neutral")
    smooth = mode_filter(labels, window)
    # smoothing may bridge flicker, but a frame with no usable face is never given an expression
    return [None if raw is None else sm for raw, sm in zip(labels, smooth)]


class FaceExpressionDetector:
    """Runs Face Mesh on a crop around the head located by the pose landmarks."""

    def __init__(self, min_detection_confidence: float = 0.5):
        solutions = load_mediapipe_solutions()
        self._mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=min_detection_confidence)

    def detect(self, frame_bgr: np.ndarray, pose: np.ndarray) -> Optional[dict]:
        h, w = frame_bgr.shape[:2]
        if pose[LM.NOSE][3] < 0.5:
            return None
        nose = pose[LM.NOSE][:2] * [w, h]
        ear_ok = pose[LM.L_EAR][3] > 0.4 and pose[LM.R_EAR][3] > 0.4
        if ear_ok:
            ear_d = _d(pose[LM.L_EAR][:2] * [w, h], pose[LM.R_EAR][:2] * [w, h])
        else:
            ear_d = 0.45 * _d(pose[LM.L_SHOULDER][:2] * [w, h], pose[LM.R_SHOULDER][:2] * [w, h])
        s = int(ear_d * 2.8)
        if s < 80:  # face too small for reliable expression analysis
            return None
        s = min(s, min(w, h))
        x0 = int(np.clip(nose[0] - s / 2, 0, w - s))
        y0 = int(np.clip(nose[1] - s / 2, 0, h - s))
        crop = frame_bgr[y0:y0 + s, x0:x0 + s]
        if s < 256:
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        res = self._mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
        return face_metrics(pts)  # square crop -> isotropic

    def close(self):
        try:
            self._mesh.close()
        except Exception:
            pass


def expression_events(times: Sequence[float], labels: Sequence[Optional[str]],
                      end_time: Optional[float] = None, min_duration: float = 1.0) -> List[dict]:
    edges = sample_edges(times, end_time)
    events = []
    for label, i0, i1 in group_runs(list(labels)):
        if label in (None, "neutral"):
            continue
        start, end = edges[i0], edges[i1 + 1]
        if end - start >= min_duration:
            events.append({"event_type": "expression", "label": label.replace("_", " "),
                           "start": start, "end": end, "detail": {"estimate": True}})
    return events


def attach_expressions(intervals: list, times: Sequence[float], labels: Sequence[Optional[str]]) -> None:
    """Add a dominant non-neutral expression estimate to each pose interval, in place."""
    for iv in intervals:
        iv.expression = None
        if iv.kind != "pose":
            continue
        sel = [labels[i] for i, t in enumerate(times) if iv.start - 1e-6 <= t < iv.end - 1e-6]
        sel = [s for s in sel if s]
        if len(sel) < 4:
            continue
        top = max(set(sel), key=sel.count)
        if top != "neutral" and sel.count(top) / len(sel) >= 0.6:
            iv.expression = top
