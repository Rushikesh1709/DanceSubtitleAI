"""DanceSubtitleAI application factory."""
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from config import Config


def create_app(overrides: dict = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    overrides = dict(overrides or {})
    if "DATA_DIR" in overrides:  # derive the other locations from a custom data folder (used by tests)
        base = Path(overrides["DATA_DIR"])
        overrides.setdefault("UPLOAD_DIR", base / "uploads")
        overrides.setdefault("PROCESSED_DIR", base / "processed")
        overrides.setdefault("OUTPUT_DIR", base / "outputs")
        overrides.setdefault("DATABASE_URL", "sqlite:///" + (base / "dance_subtitles.db").as_posix())
    app.config.update(overrides)
    for key in ("DATA_DIR", "UPLOAD_DIR", "PROCESSED_DIR", "OUTPUT_DIR"):
        app.config[key] = Path(app.config[key])
        app.config[key].mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .db import init_db
    app.extensions["db_factory"] = init_db(app.config["DATABASE_URL"])

    from .services.backend import create_default_backend
    app.config.setdefault("BACKEND_FACTORY", create_default_backend)

    from .services.pipeline import mark_interrupted_jobs
    mark_interrupted_jobs(app.extensions["db_factory"])

    from .routes import bp
    app.register_blueprint(bp)

    @app.template_filter("fmt_time")
    def fmt_time(seconds):
        if seconds is None:
            return "-"
        seconds = int(round(float(seconds)))
        return f"{seconds // 60}:{seconds % 60:02d}"

    @app.template_filter("fmt_ts")
    def fmt_ts(seconds):
        from .services.subtitle_service import format_vtt_time
        return format_vtt_time(float(seconds or 0))

    @app.template_filter("fmt_dt")
    def fmt_dt(value):
        return value.strftime("%d %b %Y, %H:%M") if isinstance(value, datetime) else "-"

    @app.errorhandler(413)
    def too_large(_e):
        flash(f"The file is larger than the {app.config['MAX_UPLOAD_MB']} MB upload limit.", "danger")
        return redirect(url_for("main.upload"))

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", title="Not found", message="That page or file does not exist."), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error occurred. Details are in the terminal window."), 500

    return app
