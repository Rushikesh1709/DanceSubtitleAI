"""Movement analysis: pose landmarks -> joint features -> stable labels -> time intervals.

All rules are transparent geometric thresholds (see `Thresholds`). Coordinates are converted to an
isotropic frame (x multiplied by width/height) so angles and distances are not distorted.
Distances are expressed in "torso lengths" so they do not depend on video resolution or zoom.
"""
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pose_service import LM
from .utils import group_runs, mode_filter, sample_edges

CORE_IDS = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST,
            LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE]


@dataclass
class Thresholds:
    min_visibility: float = 0.5
    raised_elev: float = 0.2        # wrist this far ABOVE shoulder (torso lengths) -> "raised"
    out_horiz: float = 0.75         # wrist this far sideways of shoulder + straight elbow -> "out"
    out_elbow: float = 140.0
    bent_elbow: float = 110.0       # elbow angle below this -> "bent"
    lowered_elev: float = -0.6      # wrist this far below shoulder -> arm hangs "lowered"
    knee_upright: float = 165.0     # knee angle above -> upright
    knee_slight: float = 150.0      # 150-165 slight bend
    knee_deep: float = 120.0        # below -> deep bend (low stance)
    wide_ankle: float = 0.9         # ankle separation / torso above -> wide stance
    turn_deg: float = 30.0          # change of estimated body yaw within ~1.2 s
    lateral_shift: float = 0.5      # hip-centre shift (torso lengths) within ~1.2 s
    still: float = 0.15             # speeds in torso lengths / second (divided by sensitivity)
    slow: float = 0.5
    fast: float = 1.4
    label_window_sec: float = 1.0
    min_interval_sec: float = 0.8


@dataclass
class MovementInterval:
    kind: str                       # "pose" or "no_pose"
    start: float
    end: float
    arm_left: Optional[str] = None
    arm_right: Optional[str] = None
    leg: Optional[str] = None
    motion: str = "moderate"        # still | slow | moderate | fast
    turning: bool = False
    lateral: bool = False
    reliability: float = 0.0
    intensity: float = 0.0
    repeated: bool = False          # first interval of a detected repeated pattern
    hands: Dict[str, str] = field(default_factory=dict)
    hand_changed: Dict[str, bool] = field(default_factory=dict)
    expression: Optional[str] = None

    @property
    def key(self):
        return (self.kind, self.arm_left, self.arm_right, self.leg)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MovementAnalysis:
    times: List[float]
    features: pd.DataFrame
    intervals: List[MovementInterval]
    intensity: List[Tuple[float, float]]
    events: List[dict]


# --------------------------------------------------------------------------- geometry helpers
def build_array(poses: Sequence[Optional[np.ndarray]], aspect: float, min_vis: float = 0.5) -> np.ndarray:
    """(N, 33, 2) isotropic xy array; NaN where the pose or the landmark is missing/low-confidence."""
    n = len(poses)
    arr = np.full((n, 33, 2), np.nan, dtype=np.float64)
    for i, p in enumerate(poses):
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float64)
        ok = p[:, 3] >= min_vis
        arr[i, ok, 0] = p[ok, 0] * aspect
        arr[i, ok, 1] = p[ok, 1]
    return arr


def smooth_landmarks(arr: np.ndarray, window: int = 3, max_gap: int = 2) -> np.ndarray:
    """Fill gaps of <= max_gap samples, then median-filter. Longer gaps stay NaN (never invented)."""
    n = arr.shape[0]
    if n == 0:
        return arr
    df = pd.DataFrame(arr.reshape(n, -1))
    na = df.isna().to_numpy()
    long_gap = np.zeros_like(na)
    for c in range(na.shape[1]):
        col = na[:, c]
        i = 0
        while i < n:
            if col[i]:
                j = i
                while j < n and col[j]:
                    j += 1
                if j - i > max_gap:
                    long_gap[i:j, c] = True
                i = j
            else:
                i += 1
    interp = df.interpolate(limit=max_gap, limit_area="inside")
    interp = interp.mask(pd.DataFrame(long_gap, index=df.index, columns=df.columns))
    rolled = interp.rolling(window=window, center=True, min_periods=1).median()
    rolled = rolled.where(~interp.isna())
    return rolled.to_numpy().reshape(arr.shape)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at point b (degrees) for arrays of 2-D points (N, 2)."""
    v1, v2 = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _nanmedian(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


# --------------------------------------------------------------------------- features
def compute_features(arr: np.ndarray, times: Sequence[float]) -> pd.DataFrame:
    """Per-sample numeric features from a smoothed landmark array."""
    n = arr.shape[0]
    P = lambda i: arr[:, i, :]
    l_sh, r_sh, l_hip, r_hip = P(LM.L_SHOULDER), P(LM.R_SHOULDER), P(LM.L_HIP), P(LM.R_HIP)
    mid_sh, mid_hip = (l_sh + r_sh) / 2, (l_hip + r_hip) / 2
    sh_w = np.linalg.norm(l_sh - r_sh, axis=1)
    torso_direct = np.linalg.norm(mid_sh - mid_hip, axis=1)
    # If the hips are out of frame, estimate the torso from the shoulder width (rough, lowers reliability).
    torso = np.where(np.isfinite(torso_direct), torso_direct, sh_w / 0.75)
    torso = np.where(torso > 0.04, torso, np.nan)  # dancer far too small -> unreliable
    torso_ref = _nanmedian(torso)

    f = {"time": np.asarray(times, dtype=float), "torso": torso}
    for side, (sh, el, wr, hip, kn, an) in {
        "l": (LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST, LM.L_HIP, LM.L_KNEE, LM.L_ANKLE),
        "r": (LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST, LM.R_HIP, LM.R_KNEE, LM.R_ANKLE)}.items():
        f[f"elev_{side}"] = (P(sh)[:, 1] - P(wr)[:, 1]) / torso
        f[f"horiz_{side}"] = np.abs(P(wr)[:, 0] - P(sh)[:, 0]) / torso
        f[f"elbow_{side}"] = angle_deg(P(sh), P(el), P(wr))
        f[f"knee_{side}"] = angle_deg(P(hip), P(kn), P(an))
    f["ankle_sep"] = np.abs(P(LM.L_ANKLE)[:, 0] - P(LM.R_ANKLE)[:, 0]) / torso
    f["center_x"] = mid_hip[:, 0]
    f["sw_ratio"] = sh_w / torso
    ref = np.nanpercentile(f["sw_ratio"], 90) if np.isfinite(f["sw_ratio"]).any() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        f["yaw_deg"] = np.degrees(np.arccos(np.clip(f["sw_ratio"] / ref, 0.0, 1.0)))

    # speed of wrists, ankles and hip centre in torso lengths per second
    pts = np.stack([P(LM.L_WRIST), P(LM.R_WRIST), P(LM.L_ANKLE), P(LM.R_ANKLE), mid_hip], axis=1)
    speed = np.full(n, np.nan)
    if n > 1:
        dt = np.diff(np.asarray(times, dtype=float))
        dt[dt <= 0] = np.nan
        disp = np.linalg.norm(np.diff(pts, axis=0), axis=2)          # (N-1, 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_disp = np.nanmean(disp, axis=1)
        speed[1:] = mean_disp / dt / torso_ref if np.isfinite(torso_ref) else np.nan
        speed[0] = speed[1]
    f["speed"] = pd.Series(speed).rolling(3, center=True, min_periods=1).mean().to_numpy()
    f["visible_fraction"] = np.isfinite(arr[:, CORE_IDS, 0]).mean(axis=1)
    return pd.DataFrame(f)


# --------------------------------------------------------------------------- per-sample states
def arm_state(elev, horiz, elbow, th: Thresholds) -> Optional[str]:
    if not (np.isfinite(elev) and np.isfinite(horiz) and np.isfinite(elbow)):
        return None
    if elev > th.raised_elev:
        return "raised"
    if horiz > th.out_horiz and elbow > th.out_elbow:
        return "out"
    if elbow < th.bent_elbow:
        return "bent"
    if elev < th.lowered_elev:
        return "lowered"
    return "mid"


def leg_state(knee_l, knee_r, ankle_sep, th: Thresholds) -> Optional[str]:
    knees = [k for k in (knee_l, knee_r) if np.isfinite(k)]
    if not knees:
        return None
    knee = float(np.mean(knees))
    wide = np.isfinite(ankle_sep) and ankle_sep > th.wide_ankle
    if knee >= th.knee_upright:
        return "wide_upright" if wide else "upright"
    if knee >= th.knee_slight:
        return "wide_slight" if wide else "slight"
    if knee >= th.knee_deep:
        return "wide_bent" if wide else "bent"
    return "wide_deep" if wide else "deep"


def motion_class(speed: float, sensitivity: float, th: Thresholds) -> str:
    if not np.isfinite(speed):
        return "moderate"
    s = max(sensitivity, 0.1)
    if speed < th.still / s:
        return "still"
    if speed < th.slow / s:
        return "slow"
    if speed < th.fast / s:
        return "moderate"
    return "fast"


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


# --------------------------------------------------------------------------- main entry point
def analyze_movement(times: Sequence[float], poses: Sequence[Optional[np.ndarray]], aspect: float,
                     sensitivity: float = 1.0, end_time: Optional[float] = None,
                     thresholds: Optional[Thresholds] = None) -> MovementAnalysis:
    th = thresholds or Thresholds()
    n = len(times)
    if n == 0:
        return MovementAnalysis([], pd.DataFrame(), [], [], [])
    times = [float(t) for t in times]
    edges = sample_edges(times, end_time)
    dts = np.diff(times)
    dt = float(np.median(dts[dts > 0])) if (dts > 0).any() else 0.2
    fps_s = 1.0 / dt

    arr = smooth_landmarks(build_array(poses, aspect, th.min_visibility), window=3, max_gap=2)
    feats = compute_features(arr, times)

    raw_l, raw_r, raw_leg = [], [], []
    for i in range(n):
        shoulders_ok = np.isfinite(feats["torso"].iat[i])
        a_l = arm_state(feats["elev_l"].iat[i], feats["horiz_l"].iat[i], feats["elbow_l"].iat[i], th) if shoulders_ok else None
        a_r = arm_state(feats["elev_r"].iat[i], feats["horiz_r"].iat[i], feats["elbow_r"].iat[i], th) if shoulders_ok else None
        lg = leg_state(feats["knee_l"].iat[i], feats["knee_r"].iat[i], feats["ankle_sep"].iat[i], th) if shoulders_ok else None
        if a_l is None and a_r is None:  # not enough information to describe this sample
            a_l = a_r = lg = None
        raw_l.append(a_l); raw_r.append(a_r); raw_leg.append(lg)

    win = _odd(round(th.label_window_sec * fps_s))
    sm_l, sm_r, sm_leg = mode_filter(raw_l, win), mode_filter(raw_r, win), mode_filter(raw_leg, win)
    # Smoothing may bridge only SHORT detection gaps; a long gap must stay a gap ("dancer not visible").
    max_fill = max(2, int(round(0.6 * fps_s)))
    missing = [a is None and b is None for a, b in zip(raw_l, raw_r)]
    for label, i0, i1 in group_runs(missing):
        if label and (i1 - i0 + 1) > max_fill:
            for series in (sm_l, sm_r, sm_leg):
                for i in range(i0, i1 + 1):
                    series[i] = None
    keys = [("pose", a, b, c) if (a or b) else ("no_pose", None, None, None)
            for a, b, c in zip(sm_l, sm_r, sm_leg)]

    # ---- group into runs, absorb runs that are too short
    min_samples = max(1, int(round(th.min_interval_sec * fps_s)))
    runs = [[k, i0, i1] for k, i0, i1 in group_runs(keys)]
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (k, i0, i1) in enumerate(runs):
            if (i1 - i0 + 1) < min_samples:
                tgt = j - 1 if j > 0 else j + 1
                if j > 0 and j + 1 < len(runs) and (runs[j + 1][2] - runs[j + 1][1]) > (runs[j - 1][2] - runs[j - 1][1]):
                    tgt = j + 1
                runs[tgt][1] = min(runs[tgt][1], i0)
                runs[tgt][2] = max(runs[tgt][2], i1)
                runs.pop(j)
                changed = True
                break
        # merge adjacent runs that now share the same key
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
                changed = True
            else:
                merged.append(r)
        runs = merged

    # ---- turning / lateral movement flags (per sample)
    k = max(1, int(round(0.6 * fps_s)))
    yaw, cx = feats["yaw_deg"].to_numpy(), feats["center_x"].to_numpy()
    torso_ref = _nanmedian(feats["torso"])
    turning = np.zeros(n, dtype=bool)
    lateral = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if np.isfinite(yaw[a]) and np.isfinite(yaw[b]):
            turning[i] = abs(yaw[b] - yaw[a]) > th.turn_deg
        if np.isfinite(cx[a]) and np.isfinite(cx[b]) and np.isfinite(torso_ref):
            lateral[i] = abs(cx[b] - cx[a]) / torso_ref > th.lateral_shift

    speed = feats["speed"].to_numpy()
    intervals: List[MovementInterval] = []
    for key, i0, i1 in runs:
        iv = MovementInterval(kind=key[0], start=edges[i0], end=edges[i1 + 1])
        sl = slice(i0, i1 + 1)
        if key[0] == "pose":
            iv.arm_left, iv.arm_right, iv.leg = key[1], key[2], key[3]
            sp = speed[sl]
            iv.intensity = float(np.nanmean(sp)) if np.isfinite(sp).any() else 0.0
            iv.motion = motion_class(iv.intensity, sensitivity, th)
            iv.turning = bool(turning[sl].mean() >= 0.4)
            iv.lateral = bool(lateral[sl].mean() >= 0.4)
            agree = []
            for i in range(i0, i1 + 1):
                comps = [(raw_l[i], key[1]), (raw_r[i], key[2]), (raw_leg[i], key[3])]
                comps = [(r, kk) for r, kk in comps if kk is not None]
                match = np.mean([1.0 if r == kk else 0.0 for r, kk in comps]) if comps else 0.0
                agree.append(match * float(feats["visible_fraction"].iat[i]))
            iv.reliability = float(np.mean(agree)) if agree else 0.0
        intervals.append(iv)

    _mark_repetitions(intervals)
    events = _build_events(intervals, times, turning, edges)
    intensity = [(t, float(s) if np.isfinite(s) else 0.0) for t, s in zip(times, speed)]
    feats = feats.assign(arm_left=sm_l, arm_right=sm_r, leg=sm_leg)
    return MovementAnalysis(times, feats, intervals, intensity, events)


def _mark_repetitions(intervals: List[MovementInterval]) -> List[Tuple[int, int]]:
    """Find A-B-A-B-A style alternation of poses (>= 5 intervals) and flag the first interval."""
    found = []
    i = 0
    while i < len(intervals) - 2:
        j = i
        while j + 2 < len(intervals) and intervals[j].kind == "pose" and \
                intervals[j].key == intervals[j + 2].key and intervals[j].key != intervals[j + 1].key:
            j += 1
        if j - i >= 2:  # at least A B A B A
            intervals[i].repeated = True
            found.append((i, j + 1))
            i = j + 2
        else:
            i += 1
    intervals_ranges = found
    return intervals_ranges


def describe_key_short(iv: MovementInterval) -> str:
    return f"L arm {iv.arm_left or '?'}, R arm {iv.arm_right or '?'}, legs {iv.leg or '?'}"


def _build_events(intervals, times, turning, edges) -> List[dict]:
    events = []
    for iv in intervals:
        if iv.kind == "no_pose":
            events.append({"event_type": "no_pose", "label": "dancer not clearly detected",
                           "start": iv.start, "end": iv.end, "reliability": 0.0, "detail": {}})
        else:
            events.append({"event_type": "movement", "label": describe_key_short(iv),
                           "start": iv.start, "end": iv.end, "reliability": iv.reliability,
                           "detail": {"motion": iv.motion, "turning": iv.turning, "lateral": iv.lateral,
                                      "intensity": round(iv.intensity, 3)}})
            if iv.repeated:
                events.append({"event_type": "repeated_movement", "label": "repeated alternation of postures starts",
                               "start": iv.start, "end": iv.end, "reliability": iv.reliability, "detail": {}})
    for label, i0, i1 in group_runs([bool(x) for x in turning]):
        if label and edges[i1 + 1] - edges[i0] >= 0.6:
            events.append({"event_type": "turn", "label": "body orientation changes",
                           "start": edges[i0], "end": edges[i1 + 1], "reliability": None, "detail": {}})
    return sorted(events, key=lambda e: (e["start"], e["event_type"]))
