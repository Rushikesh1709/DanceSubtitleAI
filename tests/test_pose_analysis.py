"""Mocked tests: pose-landmark processing, missing poses, hand shapes, face geometry."""
import numpy as np
import pytest

from helpers import make_hand, make_pose
from app.services.hand_gesture_service import classify_hand_shape, finger_states
from app.services.expression_service import classify_expressions
from app.services.movement_service import build_array, smooth_landmarks


@pytest.mark.mocked
def test_build_array_masks_low_visibility_and_missing_poses():
    p = make_pose()
    p[15, 3] = 0.1                                     # left wrist not visible
    arr = build_array([p, None], aspect=2.0, min_vis=0.5)
    assert arr.shape == (2, 33, 2)
    assert np.isnan(arr[0, 15]).all()                  # low-confidence landmark removed
    assert not np.isnan(arr[0, 11]).any()
    assert np.isnan(arr[1]).all()                      # missing pose -> all NaN
    assert arr[0, 11, 0] == pytest.approx(p[11, 0] * 2.0)   # x scaled to an isotropic frame


@pytest.mark.mocked
def test_smoothing_bridges_short_gaps_but_not_long_ones():
    series = [make_pose() for _ in range(12)]
    series[3] = None
    for i in range(6, 11):
        series[i] = None
    arr = smooth_landmarks(build_array(series, 1.0), window=3, max_gap=2)
    assert not np.isnan(arr[3, 11]).any()              # 1-sample dropout filled
    assert np.isnan(arr[8, 11]).all()                  # 5-sample gap NOT invented
    assert np.isnan(arr[6, 11]).all() and np.isnan(arr[10, 11]).all()


@pytest.mark.mocked
def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(30):
        p = make_pose()
        p[15, 1] += rng.normal(0, 0.02)
        poses.append(p)
    raw = build_array(poses, 1.0)[:, 15, 1]
    smooth = smooth_landmarks(build_array(poses, 1.0))[:, 15, 1]
    assert np.nanstd(np.diff(smooth)) < np.nanstd(np.diff(raw))


@pytest.mark.mocked
@pytest.mark.parametrize("fingers,thumb,spread,expected", [
    (("index", "middle", "ring", "pinky"), False, False, "flat_hand_thumb_tucked"),
    (("index", "middle", "ring", "pinky"), True, False, "open_hand_flat"),
    (("index", "middle", "ring", "pinky"), True, True, "open_hand_spread"),
    ((), False, False, "fist"),
    ((), True, False, "fist_thumb_out"),
    (("index",), False, False, "index_extended"),
    (("index", "middle"), False, False, "two_fingers_extended"),
    (("index",), True, False, "thumb_index_extended"),
])
def test_hand_shape_rules(fingers, thumb, spread, expected):
    assert classify_hand_shape(make_hand(fingers, thumb, spread)) == expected


@pytest.mark.mocked
def test_finger_states_reports_each_finger():
    s = finger_states(make_hand(("index", "pinky"), thumb=False))
    assert s["index"] and s["pinky"] and not s["middle"] and not s["ring"] and not s["thumb"]


@pytest.mark.mocked
def test_expression_classifier_needs_enough_frontal_samples():
    few = [{"frontal": True, "mouth_width": .4, "mouth_open": .05, "corner_lift": 0, "brow_gap": .1}] * 3
    assert classify_expressions(few) == [None, None, None]


@pytest.mark.mocked
def test_expression_classifier_detects_relative_smile_and_brows():
    base = {"frontal": True, "mouth_width": 0.40, "mouth_open": 0.03, "corner_lift": 0.0, "brow_gap": 0.10}
    smile = dict(base, mouth_width=0.46, corner_lift=0.03)
    brows = dict(base, brow_gap=0.13)
    series = [base] * 20 + [smile] * 8 + [base] * 6 + [brows] * 8 + [None, {"frontal": False, **{k: 0 for k in base if k != "frontal"}}]
    labels = classify_expressions(series)
    assert "smiling" in labels and "raised_eyebrows" in labels
    assert labels[0] == "neutral"
    assert labels[-1] is None and labels[-2] is None   # missing / non-frontal faces are never labelled
