"""REAL inference test: runs MediaPipe on a real dance video.

Skipped unless the classic MediaPipe 'solutions' API is installed AND the environment variable
DANCE_TEST_VIDEO points to a short video (10-30 s) of a dancer.
PowerShell:   $env:DANCE_TEST_VIDEO="C:\\videos\\bharatanatyam_clip.mp4"; python -m pytest tests/test_real_video.py -v
"""
import os

import pytest

pytestmark = pytest.mark.real_video

VIDEO = os.getenv("DANCE_TEST_VIDEO", "")


def _solutions_available():
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions")
    except ImportError:
        return False


@pytest.mark.skipif(not VIDEO or not os.path.isfile(VIDEO), reason="set DANCE_TEST_VIDEO to a real video path")
@pytest.mark.skipif(not _solutions_available(), reason="MediaPipe 'solutions' API not available")
def test_real_video_end_to_end():
    from app.services import video_service
    from app.services.backend import MediaPipeBackend
    from app.services.movement_service import analyze_movement
    from app.services.subtitle_service import generate_segments, to_srt, validate_segments

    meta = video_service.validate_video(VIDEO)
    backend = MediaPipeBackend(enable_hands=True, enable_face=False)
    times, poses = [], []
    try:
        for t, _i, frame in video_service.iter_sampled_frames(VIDEO, 4):
            times.append(t)
            poses.append(backend.analyze(frame)["pose"])
    finally:
        backend.close()
    assert sum(p is not None for p in poses) >= 0.3 * len(poses), "dancer was found in too few frames"
    analysis = analyze_movement(times, poses, meta.width / meta.height, end_time=meta.duration)
    segs = generate_segments(analysis.intervals, meta.duration, "Bharatanatyam")
    assert segs and validate_segments(segs, meta.duration) == []
    print("\n" + to_srt(segs))
