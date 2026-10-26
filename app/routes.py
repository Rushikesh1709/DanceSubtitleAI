"""All web pages and JSON endpoints."""
import json
import re
from pathlib import Path

from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify, redirect, render_template,
                   request, send_file, url_for)
from sqlalchemy import delete, select
from werkzeug.utils import secure_filename

from .models import AnalysisEvent, ProcessingJob, SubtitleSegment
from .services import export_service, pipeline, video_service
from .services.settings_service import load_settings, save_settings, validate_settings
from .services.subtitle_service import (SubtitleSettings, merge_segments, segments_from_payload, split_segment,
                                        to_srt, to_vtt, validate_segments, Segment)

bp = Blueprint("main", __name__)

EXAMPLE_SUBTITLES = [
    ("00:00:01.000", "The dancer raises both arms while standing upright."),
    ("00:00:03.600", "The dancer bends the knees and moves into a lower stance."),
    ("00:00:06.300", "The right hand holds an open, flat hand shape."),
]


# ----------------------------------------------------------------------------- helpers
def db():
    if "db" not in g:
        g.db = current_app.extensions["db_factory"]()
    return g.db


@bp.teardown_app_request
def _close_db(_exc):
    session = g.pop("db", None)
    if session is not None:
        session.close()


def get_settings() -> dict:
    return load_settings(db(), current_app.config["DEFAULT_SETTINGS"])


def get_job(public_id: str) -> ProcessingJob:
    job = db().execute(select(ProcessingJob).where(ProcessingJob.public_id == public_id)).scalar_one_or_none()
    if job is None:
        abort(404)
    return job


def _inside(path, root) -> bool:
    try:
        p, r = Path(path).resolve(), Path(root).resolve()
        return r in p.parents and p.is_file()
    except (OSError, ValueError, TypeError):
        return False


def serve_job_file(path, **kwargs):
    """Only files located inside the upload / processed folders can ever be served."""
    roots = (current_app.config["UPLOAD_DIR"], current_app.config["PROCESSED_DIR"])
    if not path or not any(_inside(path, r) for r in roots):
        abort(404)
    return send_file(path, conditional=True, **kwargs)


def segments_of(job) -> list:
    return [Segment(s.start_sec, s.end_sec, s.text, s.action_category, s.confidence) for s in job.segments]


def subtitle_settings() -> SubtitleSettings:
    return SubtitleSettings.from_dict(get_settings())


def slug(job) -> str:
    return secure_filename(job.display_title)[:60] or "subtitles"


def status_payload(job) -> dict:
    return {
        "status": job.status, "stage": job.stage, "stage_detail": job.stage_detail,
        "pose_status": job.pose_status, "gesture_status": job.gesture_status,
        "expression_status": job.expression_status, "subtitle_status": job.subtitle_status,
        "frames_done": job.frames_done, "frames_total": job.frames_total,
        "error": job.error_message, "results_url": url_for("main.results", public_id=job.public_id),
    }


# ----------------------------------------------------------------------------- pages
@bp.route("/")
def index():
    return render_template("index.html", examples=EXAMPLE_SUBTITLES,
                           formats=sorted(current_app.config["ALLOWED_EXTENSIONS"]))


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    settings = get_settings()
    cfg = current_app.config
    if request.method == "GET":
        return render_template("upload.html", settings=settings, dance_forms=cfg["DANCE_FORMS"], form={})

    form = request.form
    file = request.files.get("video")
    errors = []
    if file is None or not file.filename:
        errors.append("Please choose a video file.")
    elif not video_service.allowed_file(file.filename, cfg["ALLOWED_EXTENSIONS"]):
        errors.append("Unsupported file type. Allowed: " + ", ".join(sorted(cfg["ALLOWED_EXTENSIONS"])) + ".")

    options, setting_errors = validate_settings(
        {"sampling_fps": form.get("sampling_fps"),
         "enable_hands": "1" if form.get("enable_hands") else "0",
         "enable_face": "1" if form.get("enable_face") else "0"}, settings)
    errors += setting_errors
    dance_form = form.get("dance_form", "")
    if dance_form and dance_form not in cfg["DANCE_FORMS"]:
        errors.append("Unknown dance form.")
    if form.get("language", "en") != "en":
        errors.append("Only English subtitles are supported in this version.")
    title = (form.get("title") or "").strip()[:200]

    if errors:
        for e in errors:
            flash(e, "danger")
        return render_template("upload.html", settings=settings, dance_forms=cfg["DANCE_FORMS"], form=form), 400

    stored = video_service.make_stored_filename(file.filename)
    dest = Path(cfg["UPLOAD_DIR"]) / stored
    file.save(dest)
    try:
        meta = video_service.validate_video(dest, cfg["MAX_VIDEO_SECONDS"])
    except video_service.VideoError as exc:
        dest.unlink(missing_ok=True)
        flash(str(exc), "danger")
        return render_template("upload.html", settings=settings, dance_forms=cfg["DANCE_FORMS"], form=form), 400

    job = ProcessingJob(title=title, original_filename=secure_filename(file.filename) or "video",
                        stored_filename=stored, video_path=str(dest), dance_form=dance_form, language="en",
                        duration_sec=meta.duration, width=meta.width, height=meta.height, fps=meta.fps,
                        total_frames=meta.total_frames, options_json=json.dumps(options), status="queued")
    db().add(job)
    db().commit()
    pipeline.start_job(current_app._get_current_object(), job.id)
    return redirect(url_for("main.processing", public_id=job.public_id))


@bp.route("/jobs/<public_id>/processing")
def processing(public_id):
    job = get_job(public_id)
    if job.status == "completed":
        return redirect(url_for("main.results", public_id=public_id))
    return render_template("processing.html", job=job)


@bp.route("/jobs/<public_id>")
def results(public_id):
    job = get_job(public_id)
    if job.status != "completed":
        return redirect(url_for("main.processing", public_id=public_id))
    events = {}
    for e in job.events:
        events.setdefault(e.event_type, []).append(e)
    return render_template("results.html", job=job, summary=job.summary, events=events,
                           settings=get_settings(),
                           ffmpeg=export_service.ffmpeg_status(current_app.config.get("FFMPEG_PATH", "")))


@bp.route("/jobs/<public_id>/edit")
def editor(public_id):
    job = get_job(public_id)
    if job.status != "completed":
        return redirect(url_for("main.processing", public_id=public_id))
    return render_template("editor.html", job=job)


@bp.route("/jobs/<public_id>/visualization")
def visualization(public_id):
    job = get_job(public_id)
    if job.status != "completed":
        return redirect(url_for("main.processing", public_id=public_id))
    return render_template("visualization.html", job=job)


@bp.route("/history")
def history():
    jobs = db().execute(select(ProcessingJob).order_by(ProcessingJob.created_at.desc())).scalars().all()
    return render_template("history.html", jobs=jobs)


@bp.route("/settings", methods=["GET", "POST"])
def settings_page():
    current = get_settings()
    if request.method == "POST":
        data = {k: v for k, v in request.form.items()}
        for key in ("enable_hands", "enable_face", "show_no_pose_subtitles", "show_mudra_candidates"):
            data[key] = "1" if request.form.get(key) else "0"   # unchecked boxes are absent from forms
        clean, errors = validate_settings(data, current)
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("settings.html", settings=data, defaults=current_app.config["DEFAULT_SETTINGS"]), 400
        save_settings(db(), clean)
        flash("Settings saved. They apply to videos processed from now on (and to new downloads).", "success")
        return redirect(url_for("main.settings_page"))
    return render_template("settings.html", settings=current, defaults=current_app.config["DEFAULT_SETTINGS"])


@bp.route("/jobs/<public_id>/delete", methods=["POST"])
def delete_job(public_id):
    job = get_job(public_id)
    if job.status == "processing":
        flash("This job is still running and cannot be deleted yet.", "warning")
        return redirect(url_for("main.history"))
    pipeline.delete_job_files(current_app._get_current_object(), job)
    db().delete(job)
    db().commit()
    flash("The job and its files were deleted.", "success")
    return redirect(url_for("main.history"))


# ----------------------------------------------------------------------------- files & downloads
@bp.route("/jobs/<public_id>/video")
def video(public_id):
    job = get_job(public_id)
    return serve_job_file(job.preview_path or job.video_path)


@bp.route("/jobs/<public_id>/subtitles.vtt")
def subtitles_track(public_id):
    job = get_job(public_id)
    return Response(to_vtt(segments_of(job), subtitle_settings()), mimetype="text/vtt")


@bp.route("/jobs/<public_id>/download/<kind>")
def download(public_id, kind):
    job = get_job(public_id)
    name = slug(job)
    if kind in ("srt", "vtt"):
        s = subtitle_settings()
        body = to_srt(segments_of(job), s) if kind == "srt" else to_vtt(segments_of(job), s)
        mime = "application/x-subrip" if kind == "srt" else "text/vtt"
        return Response(body, mimetype=mime, headers={
            "Content-Disposition": f'attachment; filename="{name}.{kind}"'})
    out_dir = Path(current_app.config["PROCESSED_DIR"]) / job.public_id
    if kind == "features":
        return serve_job_file(out_dir / "features.csv", as_attachment=True, download_name=f"{name}_features.csv")
    if kind == "analysis":
        return serve_job_file(out_dir / "analysis.json", as_attachment=True, download_name=f"{name}_analysis.json")
    if kind == "video":
        if job.export_status != "done":
            abort(404)
        return serve_job_file(job.export_path, as_attachment=True, download_name=f"{name}_subtitled.mp4")
    abort(404)


@bp.route("/api/jobs/<public_id>/analysis")
def analysis_json(public_id):
    job = get_job(public_id)
    return serve_job_file(Path(current_app.config["PROCESSED_DIR"]) / job.public_id / "analysis.json")


# ----------------------------------------------------------------------------- JSON API
@bp.route("/api/jobs/<public_id>/status")
def api_status(public_id):
    return jsonify(status_payload(get_job(public_id)))


@bp.route("/api/jobs/<public_id>/segments", methods=["GET", "PUT"])
def api_segments(public_id):
    job = get_job(public_id)
    if request.method == "GET":
        return jsonify(segments=[s.to_dict() for s in job.segments], duration=job.duration_sec)
    data = request.get_json(silent=True) or {}
    segs, errors = segments_from_payload(data.get("segments"))
    if not errors:
        errors = validate_segments(segs, job.duration_sec)
    if errors:
        return jsonify(ok=False, errors=errors[:10]), 400
    db().execute(delete(SubtitleSegment).where(SubtitleSegment.job_id == job.id))
    for i, g in enumerate(segs):
        db().add(SubtitleSegment(job_id=job.id, idx=i, start_sec=round(g.start, 3), end_sec=round(g.end, 3),
                                 text=g.text, action_category=g.category, confidence=g.confidence, edited=True))
    db().commit()
    return jsonify(ok=True, count=len(segs))


@bp.route("/api/jobs/<public_id>/segments/transform", methods=["POST"])
def api_transform(public_id):
    """Stateless merge / split helper used by the editor (nothing is saved until 'Save')."""
    get_job(public_id)
    data = request.get_json(silent=True) or {}
    segs, errors = segments_from_payload(data.get("segments"))
    if errors:
        return jsonify(ok=False, errors=errors[:10]), 400
    try:
        index = int(data.get("index"))
        if data.get("op") == "merge":
            segs = merge_segments(segs, index)
        elif data.get("op") == "split":
            segs = split_segment(segs, index, float(data.get("at")))
        else:
            return jsonify(ok=False, errors=["Unknown operation."]), 400
    except (ValueError, TypeError) as exc:
        return jsonify(ok=False, errors=[str(exc) or "Invalid request."]), 400
    return jsonify(ok=True, segments=[s.to_dict() for s in segs])


@bp.route("/jobs/<public_id>/export-video", methods=["POST"])
def export_video(public_id):
    job = get_job(public_id)
    status = export_service.ffmpeg_status(current_app.config.get("FFMPEG_PATH", ""))
    if not (status["available"] and status["subtitles_filter"]):
        return jsonify(ok=False, error=status["message"]), 400
    if job.export_status == "running":
        return jsonify(ok=True, status="running")
    job.export_status, job.export_error = "running", None
    db().commit()
    pipeline.start_export(current_app._get_current_object(), job.id)
    return jsonify(ok=True, status="running")


@bp.route("/api/jobs/<public_id>/export-status")
def export_status(public_id):
    job = get_job(public_id)
    return jsonify(status=job.export_status, error=job.export_error,
                   download_url=url_for("main.download", public_id=job.public_id, kind="video")
                   if job.export_status == "done" else None)
