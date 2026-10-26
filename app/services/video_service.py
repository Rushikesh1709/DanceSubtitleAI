"""Video validation, metadata reading and timestamp-preserving frame sampling."""
import math
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

import cv2
import numpy as np
from werkzeug.utils import secure_filename


class VideoError(Exception):
    """Raised for unreadable / unsupported / corrupted video files (message is user-safe)."""


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    fourcc: str = ""
    fps_was_guessed: bool = False

    def to_dict(self):
        return asdict(self)


def allowed_file(filename: str, allowed_extensions: Set[str]) -> bool:
    return Path(filename or "").suffix.lower() in allowed_extensions


def make_stored_filename(original_filename: str) -> str:
    """Return a unique, safe filename (keeps only a validated extension)."""
    ext = Path(secure_filename(original_filename or "")).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def _fourcc_to_str(value: float) -> str:
    try:
        v = int(value)
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    except Exception:
        return ""


def read_metadata(path) -> VideoMeta:
    """Read fps, size, duration and frame count. Raises VideoError if the file cannot be opened."""
    path = str(path)
    if not os.path.isfile(path):
        raise VideoError("The video file could not be found.")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("The video could not be opened. It may be corrupted or use an unsupported codec.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        guessed = False
        if not (fps and math.isfinite(fps) and 1.0 <= fps <= 240.0):
            fps, guessed = 25.0, True
        if total <= 0:  # some containers do not store a frame count: count by scanning
            total = 0
            while cap.grab():
                total += 1
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if ok:
                height, width = frame.shape[:2]
        if width <= 0 or height <= 0 or total <= 0:
            raise VideoError("The video contains no readable frames.")
        return VideoMeta(fps=fps, total_frames=total, width=width, height=height,
                         duration=total / fps, fourcc=fourcc, fps_was_guessed=guessed)
    finally:
        cap.release()


def validate_video(path, max_seconds: Optional[float] = None) -> VideoMeta:
    """Check that the first frame can actually be decoded and the duration is acceptable."""
    meta = read_metadata(path)
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise VideoError("The first frame of the video could not be decoded (corrupted file or unsupported codec).")
    if max_seconds and meta.duration > max_seconds:
        raise VideoError(
            f"The video is {meta.duration/60:.1f} minutes long; the configured limit is {max_seconds/60:.0f} minutes.")
    return meta


def iter_sampled_frames(path, sample_fps: float = 5.0) -> Iterator[Tuple[float, int, np.ndarray]]:
    """Yield (timestamp_seconds, frame_index, frame_bgr) at roughly `sample_fps` samples per second.

    Timestamps come from the video's own clock (CAP_PROP_POS_MSEC) whenever it is available, so
    variable-frame-rate videos and dropped frames keep correct positions on the ORIGINAL timeline.
    Frames that are skipped are only grabbed (not converted), which is much cheaper than a full read.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    meta = read_metadata(path)
    interval = 1.0 / sample_fps
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("The video could not be opened.")
    try:
        idx = 0
        next_t = 0.0
        last_t = -1.0
        while True:
            if not cap.grab():
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000.0 if (pos_ms and pos_ms > 0) else idx / meta.fps
            if t < last_t:  # clock went backwards -> fall back to index based time
                t = idx / meta.fps
            last_t = t
            if t + 1e-6 >= next_t:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield t, idx, frame
                while next_t <= t + 1e-6:
                    next_t += interval
            idx += 1
    finally:
        cap.release()


def extract_frame_at(path, time_sec: float) -> Optional[np.ndarray]:
    """Return the frame closest to `time_sec` or None."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
