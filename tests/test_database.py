"""Database tests (need SQLAlchemy). Uses a temporary SQLite file."""
import json

import pytest
from sqlalchemy import select

from app.db import init_db, session_scope
from app.models import AnalysisEvent, ApplicationSetting, ProcessingJob, SubtitleSegment
from app.services.settings_service import load_settings, save_settings, validate_settings

DEFAULTS = {"sampling_fps": 5.0, "sensitivity": 1.0, "min_duration": 1.5, "max_duration": 6.0, "min_gap": 0.1,
            "max_chars_per_line": 42, "max_lines": 2, "confidence_threshold": 0.5, "low_confidence_mode": "neutral",
            "enable_hands": True, "enable_face": False, "show_no_pose_subtitles": True,
            "show_mudra_candidates": False, "output_format": "srt", "model_complexity": 1}


@pytest.fixture()
def factory(tmp_path):
    return init_db("sqlite:///" + (tmp_path / "t.db").as_posix())


def test_job_segments_and_events_roundtrip(factory):
    with session_scope(factory) as s:
        job = ProcessingJob(title="T", original_filename="a.mp4", video_path="x", options_json=json.dumps({"a": 1}))
        job.segments.append(SubtitleSegment(idx=1, start_sec=2, end_sec=3, text="b"))
        job.segments.append(SubtitleSegment(idx=0, start_sec=0, end_sec=1, text="a", confidence=0.8))
        job.events.append(AnalysisEvent(event_type="movement", label="x", start_sec=0, end_sec=1))
        s.add(job)
    with session_scope(factory) as s:
        job = s.execute(select(ProcessingJob)).scalar_one()
        assert len(job.public_id) == 16 and job.status == "queued"
        assert [g.text for g in job.segments] == ["a", "b"]          # ordered by idx
        assert job.options == {"a": 1} and job.display_title == "T"


def test_deleting_a_job_cascades(factory):
    with session_scope(factory) as s:
        job = ProcessingJob(original_filename="a.mp4", video_path="x")
        job.segments.append(SubtitleSegment(idx=0, start_sec=0, end_sec=1, text="a"))
        job.events.append(AnalysisEvent(event_type="turn", label="x", start_sec=0, end_sec=1))
        s.add(job)
    with session_scope(factory) as s:
        s.delete(s.execute(select(ProcessingJob)).scalar_one())
    with session_scope(factory) as s:
        assert s.query(SubtitleSegment).count() == 0 and s.query(AnalysisEvent).count() == 0


def test_foreign_keys_are_enforced(factory):
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        with session_scope(factory) as s:
            s.add(SubtitleSegment(job_id=999, idx=0, start_sec=0, end_sec=1, text="orphan"))


def test_settings_persist(factory):
    with session_scope(factory) as s:
        assert load_settings(s, DEFAULTS) == DEFAULTS
        clean, errors = validate_settings({"sampling_fps": "8", "enable_hands": "0", "output_format": "vtt"}, DEFAULTS)
        assert errors == []
        save_settings(s, clean)
    with session_scope(factory) as s:
        loaded = load_settings(s, DEFAULTS)
        assert loaded["sampling_fps"] == 8.0 and loaded["enable_hands"] is False and loaded["output_format"] == "vtt"
        assert s.query(ApplicationSetting).count() == len(DEFAULTS)


def test_settings_validation_rejects_bad_values():
    _, errors = validate_settings({"sampling_fps": "100", "max_duration": "1", "min_duration": "3",
                                   "low_confidence_mode": "bogus", "max_lines": "x"}, DEFAULTS)
    assert len(errors) >= 4
