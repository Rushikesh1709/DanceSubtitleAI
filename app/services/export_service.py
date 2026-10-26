"""FFmpeg helpers: browser-friendly preview copy and burned-in subtitle export."""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ExportError(Exception):
    """Raised with a short, user-safe message."""


def find_ffmpeg(configured: str = "") -> Optional[str]:
    if configured and Path(configured).is_file():
        return configured
    env = os.getenv("FFMPEG_PATH", "")
    if env and Path(env).is_file():
        return env
    return shutil.which("ffmpeg")


def ffmpeg_status(configured: str = "") -> dict:
    """{'available': bool, 'path': str|None, 'subtitles_filter': bool, 'message': str}"""
    path = find_ffmpeg(configured)
    if not path:
        return {"available": False, "path": None, "subtitles_filter": False,
                "message": "FFmpeg was not found. Install FFmpeg and add it to PATH (or set FFMPEG_PATH in .env)."}
    try:
        out = subprocess.run([path, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20)
        has_filter = " subtitles " in out.stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "path": path, "subtitles_filter": False,
                "message": "FFmpeg was found but could not be started."}
    msg = "FFmpeg is available." if has_filter else \
        "FFmpeg was found but it was built without the 'subtitles' filter (libass). Use a full build (e.g. gyan.dev 'full')."
    return {"available": True, "path": path, "subtitles_filter": has_filter, "message": msg}


def _run(cmd, cwd=None, timeout=7200):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ExportError("FFmpeg took too long and was stopped.") from exc
    except OSError as exc:
        raise ExportError("FFmpeg could not be started.") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        raise ExportError(f"FFmpeg failed: {tail[0][:200]}")


def make_browser_preview(ffmpeg: str, src: str, dst: str) -> None:
    """Re-encode to H.264/AAC MP4 so every browser can play it (original is never modified)."""
    _run([ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
          "-pix_fmt", "yuv420p", "-vf", "scale='min(1280,iw)':-2", "-c:a", "aac",
          "-movflags", "+faststart", str(dst)])


def burn_subtitles(ffmpeg: str, video: str, srt_text: str, out_path: str, work_dir: str,
                   font_size: int = 22) -> None:
    """Write a copy of the video with subtitles permanently drawn on the frames."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    srt_name = "burn_subs.srt"
    (work / srt_name).write_text(srt_text, encoding="utf-8")
    # The subtitle file is referenced by a RELATIVE name with cwd=work_dir: this avoids the
    # notorious Windows drive-letter ("C:") escaping problems in FFmpeg filter strings.
    style = f"FontSize={int(font_size)},Outline=2,Shadow=0,MarginV=24"
    vf = f"subtitles={srt_name}:force_style='{style}'"
    try:
        _run([ffmpeg, "-y", "-i", str(Path(video).resolve()), "-vf", vf, "-c:v", "libx264",
              "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
              str(Path(out_path).resolve())], cwd=str(work))
    finally:
        try:
            (work / srt_name).unlink()
        except OSError:
            pass
