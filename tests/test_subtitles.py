"""Unit tests: subtitle wording, timing rules, validation, editing helpers, SRT/VTT output."""
import re

import pytest

from helpers import make_sequence
from app.services.movement_service import MovementInterval, analyze_movement
from app.services.subtitle_service import (Segment, SubtitleSettings, format_srt_time, format_vtt_time,
                                           generate_segments, merge_segments, parse_timestamp,
                                           segments_from_payload, split_segment, to_srt, to_vtt,
                                           validate_segments, wrap_text)


def iv(start, end, l="raised", r="raised", leg="upright", motion="moderate", rel=0.9, **kw):
    return MovementInterval(kind="pose", start=start, end=end, arm_left=l, arm_right=r, leg=leg,
                            motion=motion, reliability=rel, **kw)


@pytest.mark.unit
def test_time_formatting_and_parsing():
    assert format_srt_time(3661.5) == "01:01:01,500"
    assert format_vtt_time(0.0) == "00:00:00.000"
    assert format_srt_time(59.9996) == "00:01:00,000"
    assert parse_timestamp("00:01:02,500") == pytest.approx(62.5)
    assert parse_timestamp("1:02.5") == pytest.approx(62.5)
    assert parse_timestamp("7") == 7.0
    for bad in ("", "abc", "1:2:3:4", "-3"):
        with pytest.raises(ValueError):
            parse_timestamp(bad)


@pytest.mark.unit
def test_descriptions_are_natural_language_not_raw_values():
    segs = generate_segments([iv(0, 3, "raised", "raised"), iv(3, 6, "raised", "out", leg="wide_bent")], 6.0)
    assert segs[0].text.startswith("The dancer raises both arms")
    assert "right arm" in segs[1].text and "left arm" in segs[1].text
    assert not re.search(r"\d\.\d|score|frame", segs[0].text.lower())


@pytest.mark.unit
def test_held_posture_wording():
    segs = generate_segments([iv(0, 4, "lowered", "lowered", motion="still")], 4.0)
    assert segs[0].text == "The dancer holds a balanced standing posture."
    assert segs[0].category == "held_posture"


@pytest.mark.unit
def test_araimandi_wording_only_for_bharatanatyam():
    iv1 = iv(0, 4, "out", "out", leg="wide_bent", motion="still")
    assert "araimandi" in generate_segments([iv1], 4.0, "Bharatanatyam")[0].text
    assert "araimandi" not in generate_segments([iv1], 4.0, "Kathak")[0].text


@pytest.mark.unit
def test_equal_consecutive_descriptions_are_merged():
    segs = generate_segments([iv(0, 2, motion="still", l="lowered", r="lowered"),
                              iv(2, 5, motion="still", l="lowered", r="lowered")], 5.0)
    assert len(segs) == 1 and segs[0].start == 0 and segs[0].end == 5.0


@pytest.mark.unit
def test_short_segments_are_absorbed_and_long_ones_split():
    s = SubtitleSettings(min_duration=1.5, max_duration=4.0)
    segs = generate_segments([iv(0, 5, "raised", "raised"), iv(5, 5.5, "out", "out"), iv(5.5, 6, "raised", "raised")], 6.0, None, s)
    assert all(g.duration >= 1.5 - 1e-6 for g in segs)
    long = generate_segments([iv(0, 10, "lowered", "lowered", motion="still")], 10.0, None, s)
    assert len(long) == 3 and all(g.duration <= 4.0 + 1e-6 for g in long)
    assert "continues" in long[1].text


@pytest.mark.unit
def test_gap_and_ordering_rules():
    s = SubtitleSettings(min_gap=0.2)
    segs = generate_segments([iv(0, 3, "raised", "raised"), iv(3, 6, "out", "out")], 6.0, None, s)
    assert segs[1].start - segs[0].end >= 0.2 - 1e-6
    assert validate_segments(segs, 6.0) == []


@pytest.mark.unit
def test_low_confidence_modes():
    weak = [iv(0, 3, rel=0.2)]
    neutral = generate_segments(weak, 3.0, None, SubtitleSettings(low_confidence_mode="neutral"))
    assert neutral[0].category == "uncertain" and "manual review" in neutral[0].text
    assert generate_segments(weak, 3.0, None, SubtitleSettings(low_confidence_mode="hide")) == []
    shown = generate_segments(weak, 3.0, None, SubtitleSettings(low_confidence_mode="show"))
    assert "low reliability" in shown[0].text
    borderline = generate_segments([iv(0, 3, rel=0.55)], 3.0)
    assert "(uncertain)" in borderline[0].text


@pytest.mark.unit
def test_no_pose_subtitle_can_be_disabled():
    gap = MovementInterval(kind="no_pose", start=0, end=3)
    assert "not clearly visible" in generate_segments([gap], 3.0)[0].text
    assert generate_segments([gap], 3.0, None, SubtitleSettings(show_no_pose_subtitles=False)) == []


@pytest.mark.unit
def test_hand_expression_and_mudra_candidate_wording():
    a = iv(0, 3, hands={"right": "flat_hand_thumb_tucked"}, expression="smiling")
    text = generate_segments([a], 3.0)[0].text
    assert "right hand holds a flat hand with the thumb tucked in" in text
    assert "appears to be a smile (estimate)" in text
    assert "Pataka" not in text                       # no cultural label unless explicitly enabled
    hinted = generate_segments([a], 3.0, None, SubtitleSettings(show_mudra_candidates=True))[0].text
    assert "possible resemblance to Pataka; not verified" in hinted


@pytest.mark.unit
def test_validation_catches_bad_segments():
    errs = validate_segments([Segment(2, 1, "x"), Segment(0.5, 3, ""), Segment(2.5, 4, "ok")], 3.0)
    joined = " ".join(errs)
    assert "after the start" in joined and "empty" in joined and "overlaps" in joined and "end of the video" in joined


@pytest.mark.unit
def test_payload_parsing_sorts_and_reports_errors():
    segs, errs = segments_from_payload([{"start": "00:00:05.000", "end": 7, "text": " b "},
                                        {"start": 1, "end": 2, "text": "a"}])
    assert errs == [] and [s.text for s in segs] == ["a", "b"]
    _, errs = segments_from_payload([{"start": "xx", "end": 2, "text": "a"}, {"end": 1}])
    assert len(errs) == 2
    assert segments_from_payload("nope")[1]


@pytest.mark.unit
def test_merge_and_split():
    segs = [Segment(0, 2, "The dancer raises both arms", "movement", 0.9), Segment(2, 4, "and bends the knees", "movement", 0.7)]
    merged = merge_segments(segs, 0)
    assert len(merged) == 1 and merged[0].end == 4 and merged[0].text.endswith("bends the knees")
    assert merged[0].confidence == 0.7
    with pytest.raises(ValueError):
        merge_segments(segs, 1)
    parts = split_segment(merged, 0, 2.0)
    assert len(parts) == 2 and parts[0].end < parts[1].start and (parts[0].text + " " + parts[1].text).split() == merged[0].text.split()
    assert validate_segments(parts) == []
    with pytest.raises(ValueError):
        split_segment(merged, 0, 0.1)


@pytest.mark.unit
def test_wrap_text_respects_limits_without_dropping_words():
    text = "The dancer raises both arms while standing upright, with slow, controlled movement."
    lines = wrap_text(text, 30, 2)
    assert len(lines) <= 2 and " ".join(lines) == text
    assert all(len(l) <= 60 for l in lines)


@pytest.mark.unit
def test_srt_format_is_valid():
    segs = [Segment(1.0, 3.5, "First line of text here"), Segment(3.6, 6.2, "Zweite Zeile — नमस्ते 舞")]
    srt = to_srt(segs)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines()[:2] == ["1", "00:00:01,000 --> 00:00:03,500"]
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", blocks[1].splitlines()[1])
    assert "नमस्ते 舞" in srt                           # Unicode survives


@pytest.mark.unit
def test_vtt_format_is_valid_and_escapes():
    segs = [Segment(0, 2, "Arms --> up & <out>")]
    vtt = to_vtt(segs)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.000" in vtt
    body = vtt.split("00:00:02.000\n", 1)[1]
    assert "-->" not in body and "&amp;" in body and "&lt;out>" in body


@pytest.mark.mocked
def test_end_to_end_from_poses_to_srt():
    times, poses = make_sequence([(10, {}), (15, dict(arm_l="raised", arm_r="raised")),
                                  (10, dict(arm_l="out", arm_r="out", knees="bent"))])
    a = analyze_movement(times, poses, 1.0, end_time=7.0)
    segs = generate_segments(a.intervals, 7.0, "Bharatanatyam")
    assert len(segs) >= 3 and validate_segments(segs, 7.0) == []
    assert to_srt(segs).count("-->") == len(segs)
