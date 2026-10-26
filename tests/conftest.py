"""Shared fixtures. The Flask fixtures need SQLAlchemy (installed via requirements.txt)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_pose  # noqa: E402


class FakeBackend:
    """Stands in for MediaPipe: returns scripted poses, so route/pipeline tests need no model."""

    def __init__(self, options=None):
        self.calls = 0

    def analyze(self, frame):
        self.calls += 1
        n = self.calls
        if n <= 6:
            pose = make_pose()
        elif n <= 14:
            pose = make_pose(arm_l="raised", arm_r="raised")
        else:
            pose = make_pose(arm_l="out", arm_r="out", knees="bent")
        return {"pose": pose, "hands": {"left": None, "right": None}, "face": None}

    def close(self):
        pass


@pytest.fixture()
def app(tmp_path):
    from app import create_app
    application = create_app({
        "TESTING": True, "DATA_DIR": tmp_path / "data", "AUTO_START_JOBS": False,
        "BACKEND_FACTORY": lambda options: FakeBackend(options), "MAX_VIDEO_SECONDS": 600,
    })
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()
