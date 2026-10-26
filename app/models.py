"""Database tables: processing_jobs, subtitle_segments, analysis_events, application_settings."""
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_public_id() -> str:
    return uuid.uuid4().hex[:16]


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True, default=new_public_id)
    title: Mapped[str] = mapped_column(String(200), default="")
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    stored_filename: Mapped[str] = mapped_column(String(255), default="")
    video_path: Mapped[str] = mapped_column(Text, default="")
    preview_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dance_form: Mapped[str] = mapped_column(String(60), default="")
    language: Mapped[str] = mapped_column(String(10), default="en")

    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued|processing|completed|failed
    stage: Mapped[str] = mapped_column(String(40), default="queued")
    stage_detail: Mapped[str] = mapped_column(String(255), default="")
    pose_status: Mapped[str] = mapped_column(String(15), default="pending")
    gesture_status: Mapped[str] = mapped_column(String(15), default="pending")
    expression_status: Mapped[str] = mapped_column(String(15), default="pending")
    subtitle_status: Mapped[str] = mapped_column(String(15), default="pending")
    frames_total: Mapped[int] = mapped_column(Integer, default=0)
    frames_done: Mapped[int] = mapped_column(Integer, default=0)

    options_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # shown to the user
    error_debug: Mapped[Optional[str]] = mapped_column(Text, nullable=True)     # never shown in the web UI

    export_status: Mapped[str] = mapped_column(String(15), default="none")      # none|running|done|failed
    export_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    export_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    segments: Mapped[List["SubtitleSegment"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
        order_by="SubtitleSegment.idx")
    events: Mapped[List["AnalysisEvent"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
        order_by="AnalysisEvent.start_sec")

    @property
    def options(self) -> dict:
        try:
            return json.loads(self.options_json or "{}")
        except ValueError:
            return {}

    @property
    def summary(self) -> dict:
        try:
            return json.loads(self.summary_json or "{}")
        except ValueError:
            return {}

    @property
    def display_title(self) -> str:
        return self.title or self.original_filename or f"Job {self.public_id}"


class SubtitleSegment(Base):
    __tablename__ = "subtitle_segments"
    __table_args__ = (Index("ix_segments_job_start", "job_id", "start_sec"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("processing_jobs.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    action_category: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # reliability, not a probability
    edited: Mapped[bool] = mapped_column(Boolean, default=False)

    job: Mapped[ProcessingJob] = relationship(back_populates="segments")

    def to_dict(self) -> dict:
        return {"start": self.start_sec, "end": self.end_sec, "text": self.text,
                "category": self.action_category, "confidence": self.confidence}


class AnalysisEvent(Base):
    __tablename__ = "analysis_events"
    __table_args__ = (Index("ix_events_job_type", "job_id", "event_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("processing_jobs.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(30))   # movement|gesture|expression|turn|no_pose|repeated_movement
    label: Mapped[str] = mapped_column(String(255))
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    reliability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detail_json: Mapped[str] = mapped_column(Text, default="{}")

    job: Mapped[ProcessingJob] = relationship(back_populates="events")


class ApplicationSetting(Base):
    __tablename__ = "application_settings"

    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, default="null")
