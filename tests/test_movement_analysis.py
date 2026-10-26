"""Mocked tests: movement features, state rules, temporal smoothing, interval grouping."""
import numpy as np
import pytest

from helpers import make_pose, make_sequence
from app.services.movement_service import (Thresholds, analyze_movement, arm_state, leg_state,
                                           motion_class)
from app.services.utils import group_runs, mode_filter


@pytest.mark.mocked
def test_arm_state_rules():
    th = Thresholds()
    assert arm_state(0.5, 0.1, 170, th) == "raised"
    assert arm_state(0.0, 1.0, 170, th) == "out"
    assert arm_state(-0.2, 0.2, 80, th) == "bent"
    assert arm_state(-1.0, 0.05, 170, th) == "lowered"
    assert arm_state(-0.3, 0.2, 130, th) == "mid"
    assert arm_state(float("nan"), 0.1, 170, th) is None


@pytest.mark.mocked
def test_leg_state_rules():
    th = Thresholds()
    assert leg_state(175, 176, 0.4, th) == "upright"
    assert leg_state(110, 115, 1.2, th) == "wide_deep"
    assert leg_state(135, 140, 1.2, th) == "wide_bent"
    assert leg_state(float("nan"), float("nan"), 0.4, th) is None


@pytest.mark.unit
def test_mode_filter_removes_flicker_and_keeps_gaps():
    labels = ["a"] * 5 + ["b"] + ["a"] * 5
    assert mode_filter(labels, 5) == ["a"] * 11
    assert mode_filter([None] * 5, 3) == [None] * 5
    assert group_runs(["a", "a", "b"]) == [("a", 0, 1), ("b", 2, 2)]


@pytest.mark.unit
def test_motion_class_depends_on_sensitivity():
    th = Thresholds()
    assert motion_class(0.05, 1.0, th) == "still"
    assert motion_class(1.0, 1.0, th) == "moderate"
    assert motion_class(1.0, 2.0, th) == "fast"       # same speed, higher sensitivity


@pytest.mark.mocked
def test_intervals_follow_the_scripted_postures():
    times, poses = make_sequence([(10, {}), (15, dict(arm_l="raised", arm_r="raised")),
                                  (10, dict(arm_l="out", arm_r="out", knees="bent"))])
    a = analyze_movement(times, poses, aspect=1.0, end_time=7.0)
    keys = [(iv.arm_left, iv.arm_right, iv.leg) for iv in a.intervals]
    assert keys == [("lowered", "lowered", "upright"), ("raised", "raised", "upright"), ("out", "out", "bent")]
    assert a.intervals[0].start == 0.0 and a.intervals[-1].end == pytest.approx(7.0, abs=0.01)
    assert all(b.start == pytest.approx(a_.end) for a_, b in zip(a.intervals, a.intervals[1:]))  # contiguous
    assert all(iv.reliability > 0.8 for iv in a.intervals)


@pytest.mark.mocked
def test_single_frame_glitch_does_not_create_an_interval():
    blocks = [(10, {}), (1, dict(arm_l="raised", arm_r="raised")), (10, {})]
    times, poses = make_sequence(blocks)
    a = analyze_movement(times, poses, 1.0)
    assert len(a.intervals) == 1


@pytest.mark.mocked
def test_long_detection_gap_becomes_no_pose_interval_and_short_gap_is_bridged():
    times, poses = make_sequence([(10, {}), (8, None), (10, {})])
    poses[4] = None                                    # 1-sample dropout inside the first block
    a = analyze_movement(times, poses, 1.0)
    kinds = [iv.kind for iv in a.intervals]
    assert kinds == ["pose", "no_pose", "pose"]
    gap = a.intervals[1]
    assert gap.duration == pytest.approx(1.6, abs=0.01)
    assert any(e["event_type"] == "no_pose" for e in a.events)


@pytest.mark.mocked
def test_all_missing_poses_give_one_no_pose_interval():
    times, poses = make_sequence([(10, None)])
    a = analyze_movement(times, poses, 1.0)
    assert [iv.kind for iv in a.intervals] == ["no_pose"]


@pytest.mark.mocked
def test_speed_and_motion_for_a_moving_dancer():
    blocks = [(1, dict(cx=0.20 + 0.03 * i)) for i in range(25)]
    times, poses = make_sequence(blocks)
    a = analyze_movement(times, poses, 1.0)
    assert a.features["speed"].iloc[5:].mean() > 0.3
    assert any(iv.lateral for iv in a.intervals)       # steady sideways shift detected


@pytest.mark.mocked
def test_turning_detected_when_shoulder_width_shrinks():
    blocks = [(5, dict(yaw_scale=1.0))] + [(1, dict(yaw_scale=1.0 - 0.09 * i)) for i in range(1, 9)] + [(5, dict(yaw_scale=0.25))]
    times, poses = make_sequence(blocks)
    a = analyze_movement(times, poses, 1.0)
    assert any(e["event_type"] == "turn" for e in a.events)


@pytest.mark.mocked
def test_repetition_of_alternating_postures_is_flagged():
    blocks = []
    for _ in range(3):
        blocks += [(6, dict(arm_l="raised", arm_r="raised")), (6, {})]
    blocks += [(6, dict(arm_l="raised", arm_r="raised"))]
    times, poses = make_sequence(blocks)
    a = analyze_movement(times, poses, 1.0)
    assert any(iv.repeated for iv in a.intervals)
    assert any(e["event_type"] == "repeated_movement" for e in a.events)


@pytest.mark.mocked
def test_empty_input():
    a = analyze_movement([], [], 1.0)
    assert a.intervals == [] and a.events == []
