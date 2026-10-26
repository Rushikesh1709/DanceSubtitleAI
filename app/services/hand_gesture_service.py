"""Rule-based hand-shape analysis on MediaPipe Hands landmarks.

Honest scope: this module measures WHICH FINGERS ARE EXTENDED and describes the resulting hand
shape. It does not recognise the full mudra vocabulary and never assigns symbolic meaning.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .pose_service import LM, load_mediapipe_solutions
from .utils import group_runs, mode_filter, sample_edges

VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "gesture_vocabulary.json"


def load_vocabulary(path: Path = VOCAB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _d(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def finger_states(lm: np.ndarray) -> Dict[str, bool]:
    """Which fingers are extended? `lm` is a (21, >=2) array in an isotropic coordinate frame."""
    wrist = lm[0]
    states = {}
    for name, (pip, tip) in {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}.items():
        states[name] = _d(lm[tip], wrist) > 1.12 * _d(lm[pip], wrist)
    palm_w = max(_d(lm[5], lm[17]), 1e-6)
    states["thumb"] = _d(lm[4], lm[17]) > 1.08 * _d(lm[3], lm[17]) and _d(lm[4], lm[5]) > 0.55 * palm_w
    return states


def _spread_deg(lm: np.ndarray, extended: Sequence[str]) -> float:
    """Mean angle (degrees) between neighbouring extended fingers (0 = fingers together)."""
    mcp = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    tip = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    order = [f for f in ("index", "middle", "ring", "pinky") if f in extended]
    if len(order) < 2:
        return 0.0
    angles = []
    for a, b in zip(order[:-1], order[1:]):
        va = lm[tip[a]][:2] - lm[mcp[a]][:2]
        vb = lm[tip[b]][:2] - lm[mcp[b]][:2]
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-6
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / denom, -1, 1)))))
    return float(np.mean(angles))


def classify_hand_shape(lm: np.ndarray) -> Optional[str]:
    """Return a hand-shape label (see gesture_vocabulary.json) or None if not distinguishable."""
    s = finger_states(lm)
    ext = [f for f in ("index", "middle", "ring", "pinky") if s[f]]
    n, thumb = len(ext), s["thumb"]
    if n == 4:
        if not thumb:
            return "flat_hand_thumb_tucked"
        return "open_hand_spread" if _spread_deg(lm, ext) > 14 else "open_hand_flat"
    if n == 0:
        return "fist_thumb_out" if thumb else "fist"
    if ext == ["index"] and not thumb:
        return "index_extended"
    if ext == ["index", "middle"]:
        return "two_fingers_extended"
    if ext == ["index"] and thumb:
        return "thumb_index_extended"
    if ext == ["index", "middle", "ring"]:
        return "three_fingers_extended"
    if ext == ["index", "pinky"]:
        return "index_pinky_extended"
    return None


class HandShapeDetector:
    """Detects each hand in a crop around the pose wrist (hands are tiny in full dance frames)."""

    def __init__(self, min_detection_confidence: float = 0.5):
        solutions = load_mediapipe_solutions()
        self._hands = solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, model_complexity=1,
            min_detection_confidence=min_detection_confidence)

    def _crop_box(self, pose, side, w, h):
        ids = (LM.L_WRIST, LM.L_PINKY, LM.L_INDEX, LM.L_THUMB) if side == "left" else \
              (LM.R_WRIST, LM.R_PINKY, LM.R_INDEX, LM.R_THUMB)
        pts = pose[list(ids)]
        if pts[0][3] < 0.4:
            return None
        xy = pts[:, :2] * np.array([w, h])
        center = xy.mean(axis=0)
        extent = max(np.max(np.linalg.norm(xy - center, axis=1)) * 2.0, 1.0)
        torso_px = np.linalg.norm((pose[LM.L_SHOULDER][:2] + pose[LM.R_SHOULDER][:2]) / 2 * [w, h] -
                                  (pose[LM.L_HIP][:2] + pose[LM.R_HIP][:2]) / 2 * [w, h])
        s = int(max(extent * 2.2, 0.55 * torso_px, 96))
        s = min(s, min(w, h))
        if s < 48:
            return None
        x0 = int(np.clip(center[0] - s / 2, 0, w - s))
        y0 = int(np.clip(center[1] - s / 2, 0, h - s))
        return x0, y0, s

    def detect(self, frame_bgr: np.ndarray, pose: np.ndarray) -> Dict[str, Optional[dict]]:
        h, w = frame_bgr.shape[:2]
        out: Dict[str, Optional[dict]] = {"left": None, "right": None}
        for side in ("left", "right"):
            box = self._crop_box(pose, side, w, h)
            if box is None:
                continue
            x0, y0, s = box
            crop = frame_bgr[y0:y0 + s, x0:x0 + s]
            if s < 256:
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                continue
            pts = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark], dtype=np.float32)
            shape = classify_hand_shape(pts)  # crop is square -> coordinates are isotropic
            full = np.stack([(x0 + pts[:, 0] * s) / w, (y0 + pts[:, 1] * s) / h], axis=1)
            out[side] = {"shape": shape, "landmarks": np.round(full, 3).tolist()}
        return out

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass


def hand_events(times: Sequence[float], hands: Sequence[dict], end_time: Optional[float] = None,
                min_duration: float = 1.0, window: int = 5) -> List[dict]:
    """Stable hand-shape periods (per hand) as analysis events."""
    events = []
    edges = sample_edges(times, end_time)
    for side in ("left", "right"):
        labels = [(h.get(side) or {}).get("shape") if h else None for h in hands]
        smooth = mode_filter(labels, window)
        for label, i0, i1 in group_runs(smooth):
            if label is None:
                continue
            start, end = edges[i0], edges[i1 + 1]
            if end - start >= min_duration:
                events.append({"event_type": "gesture", "label": f"{side} hand: {label}",
                               "start": start, "end": end, "detail": {"side": side, "shape": label}})
    return sorted(events, key=lambda e: e["start"])


def attach_hand_observations(intervals: list, times: Sequence[float], hands: Sequence[dict]) -> None:
    """Add dominant hand shapes (and shape changes) to each pose interval, in place."""
    for iv in intervals:
        iv.hands, iv.hand_changed = {}, {}
        if iv.kind != "pose":
            continue
        idx = [i for i, t in enumerate(times) if iv.start - 1e-6 <= t < iv.end - 1e-6]
        for side in ("left", "right"):
            shapes = [(hands[i].get(side) or {}).get("shape") for i in idx if hands[i]]
            shapes = [s for s in shapes if s]
            if len(shapes) < 3 or len(shapes) < 0.35 * max(1, len(idx)):
                continue
            counts = {s: shapes.count(s) for s in set(shapes)}
            top, top_n = max(counts.items(), key=lambda kv: kv[1])
            significant = [s for s, c in counts.items() if c / len(shapes) >= 0.25]
            if top_n / len(shapes) >= 0.6:
                iv.hands[side] = top
            if len(significant) >= 2:
                iv.hand_changed[side] = True
