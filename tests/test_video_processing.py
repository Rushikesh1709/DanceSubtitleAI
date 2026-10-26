"""Unit tests (real OpenCV on a generated video): validation, metadata, frame sampling."""
import pytest

from helpers import make_video
from app.services import video_service as vs

ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def test_allowed_file_extensions():
    assert vs.allowed_file("dance.MP4", ALLOWED)
    assert vs.allowed_file("a.webm", ALLOWED)
    assert not vs.allowed_file("script.exe", ALLOWED)
    assert not vs.allowed_file("noextension", ALLOWED)
    assert not vs.allowed_file("", ALLOWED)


def test_stored_filename_is_unique_and_safe():
    a = vs.make_stored_filename("../../etc/passwd.mp4")
    b = vs.make_stored_filename("../../etc/passwd.mp4")
    assert a != b
    assert a.endswith(".mp4") and "/" not in a and ".." not in a


def test_read_metadata(tmp_path):
    path = make_video(tmp_path / "v.mp4", n_frames=50, fps=10, size=(160, 120))
    meta = vs.read_metadata(path)
    assert (meta.width, meta.height) == (160, 120)
    assert meta.fps == pytest.approx(10, abs=0.5)
    assert meta.total_frames == 50
    assert meta.duration == pytest.approx(5.0, abs=0.3)


def test_validate_video_ok_and_too_long(tmp_path):
    path = make_video(tmp_path / "v.mp4", n_frames=50, fps=10)
    assert vs.validate_video(path).total_frames == 50
    with pytest.raises(vs.VideoError):
        vs.validate_video(path, max_seconds=2)


def test_corrupted_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is definitely not a video" * 100)
    with pytest.raises(vs.VideoError):
        vs.validate_video(bad)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(vs.VideoError):
        vs.read_metadata(tmp_path / "nope.mp4")


def test_sampling_keeps_original_timestamps(tmp_path):
    path = make_video(tmp_path / "v.mp4", n_frames=100, fps=10)   # 10 s video
    frames = list(vs.iter_sampled_frames(path, sample_fps=2))
    times = [t for t, _i, _f in frames]
    assert 18 <= len(frames) <= 21                    # about 2 samples per second
    assert times[0] == pytest.approx(0.0, abs=0.01)
    assert all(b > a for a, b in zip(times, times[1:]))  # strictly increasing
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(0.4 <= g <= 0.6 for g in gaps)
    assert times[-1] <= 10.0
    assert frames[0][2].shape == (120, 160, 3)


def test_invalid_sampling_rate(tmp_path):
    path = make_video(tmp_path / "v.mp4", n_frames=10)
    with pytest.raises(ValueError):
        list(vs.iter_sampled_frames(path, sample_fps=0))


def test_extract_frame_at(tmp_path):
    path = make_video(tmp_path / "v.mp4", n_frames=50, fps=10)
    frame = vs.extract_frame_at(path, 2.0)
    assert frame is not None and frame.shape[:2] == (120, 160)
