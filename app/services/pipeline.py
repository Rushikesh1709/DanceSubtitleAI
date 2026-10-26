"""Background processing pipeline: video -> poses -> movement -> subtitles -> stored results."""
import json
import math
import shutil
import threading
import traceback
from pathlib import Path

import numpy as np
from sqlalchemy import delete

from ..models import AnalysisEvent, ProcessingJob, SubtitleSegment, utcnow
from . import export_service, video_service
from .expression_service import attach_expressions, classify_expressions, expression_events
from .hand_gesture_service import attach_hand_observations, hand_events
from .movement_service import analyze_movement
from .pose_service import PoseBackendError
from .subtitle_service import SubtitleSettings, generate_segments, to_srt

_SLOT = threading.Semaphore(1)   # one video is analysed at a time (keeps the PC responsive)


class PipelineError(Exception):
    """An expected failure with a message that is safe to show to the user."""


def job_dir(app, public_id: str) -> Path:
    d = Path(app.config["PROCESSED_DIR"]) / public_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_job(app, job_id: int) -> None:
    if app.config.get("AUTO_START_JOBS", True):
        threading.Thread(target=run_job, args=(app, job_id), daemon=True, name=f"job-{job_id}").start()


def mark_interrupted_jobs(factory) -> int:
    """Jobs that were running when the app stopped can never finish: mark them as failed."""
    session = factory()
    try:
        stuck = session.query(ProcessingJob).filter(ProcessingJob.status.in_(["queued", "processing"])).all()
        for job in stuck:
            job.status, job.stage = "failed", "failed"
            job.error_message = "Processing was interrupted because the application was closed. Please upload the video again."
            job.finished_at = utcnow()
        for job in session.query(ProcessingJob).filter(ProcessingJob.export_status == "running").all():
            job.export_status, job.export_error = "failed", "Export was interrupted because the application was closed."
        session.commit()
        return len(stuck)
    finally:
        session.close()


def _set(session, job, **kw):
    for k, v in kw.items():
        setattr(job, k, v)
    session.commit()


def run_job(app, job_id: int) -> None:
    factory = app.extensions["db_factory"]
    with _SLOT:
        session = factory()
        try:
            _run(app, session, job_id)
        finally:
            session.close()


def _run(app, session, job_id: int) -> None:
    job = session.get(ProcessingJob, job_id)
    if job is None or job.status != "queued":
        return
    backend = None
    try:
        opts = job.options
        _set(session, job, status="processing", started_at=utcnow(), stage="validating",
             stage_detail="Checking the video file", pose_status="pending",
             gesture_status="pending" if opts.get("enable_hands") else "skipped",
             expression_status="pending" if opts.get("enable_face") else "skipped",
             subtitle_status="pending", frames_done=0)

        meta = video_service.validate_video(job.video_path, app.config["MAX_VIDEO_SECONDS"])
        _set(session, job, duration_sec=meta.duration, width=meta.width, height=meta.height,
             fps=meta.fps, total_frames=meta.total_frames)

        sample_fps = float(opts.get("sampling_fps", 5.0))
        est_total = max(1, int(meta.duration * sample_fps) + 1)
        _set(session, job, stage="loading_models", stage_detail="Loading pose model", frames_total=est_total)
        factory = app.config["BACKEND_FACTORY"]
        backend = factory(opts)

        _set(session, job, stage="extracting_poses", pose_status="running",
             gesture_status="running" if opts.get("enable_hands") else "skipped",
             expression_status="running" if opts.get("enable_face") else "skipped",
             stage_detail="Detecting body landmarks")
        times, poses, hands_obs, faces, errors = [], [], [], [], 0
        for t, _idx, frame in video_service.iter_sampled_frames(job.video_path, sample_fps):
            try:
                res = backend.analyze(frame)
            except Exception:  # one bad frame must not stop the whole video
                app.logger.exception("Frame analysis failed at %.2fs", t)
                res, errors = {"pose": None, "hands": {}, "face": None}, errors + 1
            times.append(t)
            poses.append(res.get("pose"))
            hands_obs.append(res.get("hands") or {})
            faces.append(res.get("face"))
            if len(times) % 10 == 0:
                _set(session, job, frames_done=len(times),
                     stage_detail=f"Analysed {len(times)} of about {est_total} sampled frames")
        _set(session, job, frames_done=len(times), pose_status="done",
             gesture_status="done" if opts.get("enable_hands") else "skipped",
             expression_status="done" if opts.get("enable_face") else "skipped")
        if not times:
            raise PipelineError("No frames could be read from the video.")
        if errors > 0.5 * len(times):
            raise PipelineError("The pose model failed on most frames. Check the installation (see README troubleshooting).")
        detected = sum(p is not None for p in poses)
        if detected == 0:
            raise PipelineError("No person was detected in any analysed frame. Try a video where the dancer is "
                                "larger in the frame, better lit, or increase the sampling frequency.")

        _set(session, job, stage="analyzing_movement", stage_detail="Analysing joint angles and movement")
        aspect = meta.width / float(meta.height)
        analysis = analyze_movement(times, poses, aspect, float(opts.get("sensitivity", 1.0)),
                                    end_time=meta.duration)
        events = list(analysis.events)
        if opts.get("enable_hands"):
            attach_hand_observations(analysis.intervals, times, hands_obs)
            events += hand_events(times, hands_obs, meta.duration)
        if opts.get("enable_face"):
            labels = classify_expressions([f for f in faces])
            attach_expressions(analysis.intervals, times, labels)
            events += expression_events(times, labels, meta.duration)

        _set(session, job, stage="generating_subtitles", subtitle_status="running",
             stage_detail="Writing descriptive subtitles")
        segments = generate_segments(analysis.intervals, meta.duration, job.dance_form,
                                     SubtitleSettings.from_dict(opts))

        # ---- persist results
        session.execute(delete(SubtitleSegment).where(SubtitleSegment.job_id == job.id))
        session.execute(delete(AnalysisEvent).where(AnalysisEvent.job_id == job.id))
        for i, g in enumerate(segments):
            session.add(SubtitleSegment(job_id=job.id, idx=i, start_sec=g.start, end_sec=g.end, text=g.text,
                                        action_category=g.category, confidence=g.confidence))
        for e in sorted(events, key=lambda e: e["start"]):
            session.add(AnalysisEvent(job_id=job.id, event_type=e["event_type"], label=e["label"][:255],
                                      start_sec=e["start"], end_sec=e["end"], reliability=e.get("reliability"),
                                      detail_json=json.dumps(e.get("detail", {}))))
        out_dir = job_dir(app, job.public_id)
        _write_analysis_files(out_dir, meta, sample_fps, times, poses, hands_obs, analysis, events)

        pose_iv = [iv for iv in analysis.intervals if iv.kind == "pose"]
        summary = {
            "sampled_frames": len(times), "pose_detected_frames": int(detected),
            "pose_detection_rate": round(detected / len(times), 3), "frame_errors": errors,
            "sampling_fps": sample_fps, "segments": len(segments),
            "mean_reliability": round(float(np.mean([iv.reliability for iv in pose_iv])), 3) if pose_iv else 0.0,
            "no_pose_seconds": round(sum(iv.duration for iv in analysis.intervals if iv.kind == "no_pose"), 1),
            "hands_enabled": bool(opts.get("enable_hands")), "face_enabled": bool(opts.get("enable_face")),
            "hand_frames": sum(1 for h in hands_obs if any(h.values())) if opts.get("enable_hands") else 0,
            "mean_intensity": round(float(np.mean([v for _, v in analysis.intensity])), 3),
            "fps_was_guessed": meta.fps_was_guessed,
        }
        _set(session, job, subtitle_status="done", summary_json=json.dumps(summary))

        _prepare_preview(app, session, job, meta, out_dir, summary)
        _set(session, job, status="completed", stage="completed", stage_detail="Finished",
             finished_at=utcnow())
    except (PipelineError, video_service.VideoError, PoseBackendError) as exc:
        _fail(session, job_id, str(exc), traceback.format_exc())
    except Exception:
        app.logger.exception("Unexpected error while processing job %s", job_id)
        _fail(session, job_id, "An unexpected error occurred while processing the video. "
                               "Check the terminal window for details.", traceback.format_exc())
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def _prepare_preview(app, session, job, meta, out_dir: Path, summary: dict) -> None:
    """Create an H.264 MP4 copy if the browser may not be able to play the original."""
    ext = Path(job.video_path).suffix.lower()
    needs = ext not in app.config["BROWSER_PLAYABLE_EXTENSIONS"] or meta.fourcc.lower() in ("hevc", "hvc1", "hev1")
    if not needs:
        return
    _set(session, job, stage="preparing_preview", stage_detail="Creating a browser-friendly preview copy")
    ffmpeg = export_service.find_ffmpeg(app.config.get("FFMPEG_PATH", ""))
    if not ffmpeg:
        summary["preview_note"] = ("This video format may not play in your browser and FFmpeg was not found "
                                   "to create a preview copy. Subtitles are still generated correctly.")
    else:
        dst = out_dir / "preview.mp4"
        try:
            export_service.make_browser_preview(ffmpeg, job.video_path, str(dst))
            job.preview_path = str(dst)
        except export_service.ExportError as exc:
            summary["preview_note"] = f"Preview copy could not be created ({exc})."
    job.summary_json = json.dumps(summary)
    session.commit()


def _write_analysis_files(out_dir: Path, meta, sample_fps, times, poses, hands_obs, analysis, events) -> None:
    samples = []
    for t, p, h in zip(times, poses, hands_obs):
        s = {"t": round(float(t), 3),
             "pose": None if p is None else [[round(float(x), 3), round(float(y), 3), round(float(v), 2)]
                                             for x, y, _z, v in p]}
        if h:
            hh = {side: (h[side]["landmarks"] if h.get(side) else None) for side in ("left", "right")}
            if any(hh.values()):
                s["hands"] = hh
        samples.append(s)
    payload = {
        "meta": {"duration": meta.duration, "width": meta.width, "height": meta.height,
                 "fps": meta.fps, "sample_fps": sample_fps},
        "samples": samples,
        "intensity": [[round(t, 3), round(v, 3)] for t, v in analysis.intensity],
        "intervals": [{"start": iv.start, "end": iv.end, "kind": iv.kind} for iv in analysis.intervals],
        "events": [{"type": e["event_type"], "label": e["label"], "start": round(e["start"], 3),
                    "end": round(e["end"], 3)} for e in events],
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload), encoding="utf-8")
    if not analysis.features.empty:
        analysis.features.round(4).to_csv(out_dir / "features.csv", index=False)


def _fail(session, job_id: int, message: str, debug: str) -> None:
    session.rollback()
    job = session.get(ProcessingJob, job_id)
    if job is None:
        return
    for field in ("pose_status", "gesture_status", "expression_status", "subtitle_status"):
        if getattr(job, field) == "running":
            setattr(job, field, "failed")
    _set(session, job, status="failed", stage="failed", stage_detail="", error_message=message,
         error_debug=debug[-4000:], finished_at=utcnow())


# ------------------------------------------------------------------ embedded-subtitle export
def start_export(app, job_id: int) -> None:
    threading.Thread(target=run_export, args=(app, job_id), daemon=True, name=f"export-{job_id}").start()


def run_export(app, job_id: int) -> None:
    factory = app.extensions["db_factory"]
    session = factory()
    try:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            return
        try:
            status = export_service.ffmpeg_status(app.config.get("FFMPEG_PATH", ""))
            if not status["available"] or not status["subtitles_filter"]:
                raise export_service.ExportError(status["message"])
            from .settings_service import load_settings
            settings = load_settings(session, app.config["DEFAULT_SETTINGS"])
            srt = to_srt([_seg(s) for s in job.segments], SubtitleSettings.from_dict(settings))
            out_dir = job_dir(app, job.public_id)
            dst = out_dir / "video_with_subtitles.mp4"
            export_service.burn_subtitles(status["path"], job.video_path, srt, str(dst), str(out_dir))
            _set(session, job, export_status="done", export_path=str(dst), export_error=None)
        except export_service.ExportError as exc:
            _set(session, job, export_status="failed", export_error=str(exc))
        except Exception:
            app.logger.exception("Export failed for job %s", job_id)
            _set(session, job, export_status="failed", export_error="Export failed unexpectedly.")
    finally:
        session.close()


def _seg(row):
    from .subtitle_service import Segment
    return Segment(row.start_sec, row.end_sec, row.text, row.action_category, row.confidence)


def delete_job_files(app, job: ProcessingJob) -> None:
    """Remove the uploaded video and all processed output of a job (paths come from the database only)."""
    try:
        up = Path(job.video_path).resolve()
        if Path(app.config["UPLOAD_DIR"]).resolve() in up.parents and up.is_file():
            up.unlink()
    except OSError:
        pass
    d = Path(app.config["PROCESSED_DIR"]) / job.public_id
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
