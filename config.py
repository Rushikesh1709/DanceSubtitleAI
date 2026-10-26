"""Central configuration for DanceSubtitleAI."""
import os
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent

# Default values for every user-changeable setting (stored in the database once saved).
DEFAULT_SETTINGS = {
    "sampling_fps": 5.0,            # analysed frames per second of video
    "sensitivity": 1.0,             # movement sensitivity (0.5 = less, 2.0 = more)
    "min_duration": 1.5,            # seconds, minimum subtitle length
    "max_duration": 6.0,            # seconds, maximum subtitle length
    "min_gap": 0.1,                 # seconds between two subtitles
    "max_chars_per_line": 42,
    "max_lines": 2,
    "confidence_threshold": 0.5,    # reliability below this -> neutral/hidden subtitle
    "low_confidence_mode": "neutral",  # neutral | hide | show
    "enable_hands": True,
    "enable_face": False,           # slower, off by default
    "show_no_pose_subtitles": True,
    "show_mudra_candidates": False,  # show "possible resemblance" hints (unverified)
    "output_format": "srt",         # srt | vtt (format of the main download button)
    "model_complexity": 1,          # MediaPipe pose model: 0 fast, 1 balanced, 2 accurate
}

DANCE_FORMS = [
    "Bharatanatyam", "Kathak", "Kuchipudi", "Odissi",
    "Kathakali", "Manipuri", "Mohiniyattam", "Sattriya", "Not sure / other",
]


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    UPLOAD_DIR = DATA_DIR / "uploads"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = DATA_DIR / "outputs"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///" + (DATA_DIR / "dance_subtitles.db").as_posix())

    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "1200"))
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    BROWSER_PLAYABLE_EXTENSIONS = {".mp4", ".webm"}

    DEFAULT_SETTINGS = DEFAULT_SETTINGS
    DANCE_FORMS = DANCE_FORMS
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")

    # Set to False in tests to avoid starting background threads.
    AUTO_START_JOBS = True
