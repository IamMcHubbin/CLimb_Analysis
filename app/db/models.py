"""SQLAlchemy ORM models.

These types are an implementation detail of the SQLAlchemy repository. Nothing
outside ``app.db`` should import them - routes and workers deal in the domain
records defined in ``app.db.repository``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VideoRow(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # pending until normalisation finishes, then ready or failed. Every column
    # below is null while pending - none of it is known until ffmpeg has run.
    status: Mapped[str] = mapped_column(String(16), index=True, default="pending")
    ingest_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The raw upload, kept only until it has been normalised.
    upload_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # The normalised video: constant fps, rotation baked in, long edge capped.
    # Path is relative to the data directory. The original is never kept.
    stored_path: Mapped[str] = mapped_column(String(512))
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # What the upload looked like before normalisation. Kept for debugging
    # ingest problems - orientation and frame timing bugs are invisible
    # afterwards, so the evidence has to be recorded here.
    source_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_rotation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_variable_frame_rate: Mapped[bool | None] = mapped_column(Integer, nullable=True)

    # The most recently completed run's keypoints, kept as a convenience for
    # callers that just want "the latest result". NOT the canonical location -
    # AnalysisRunRow.keypoints_path is, and a video may have several runs.
    keypoints_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    analysis_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # When the video file itself was removed. The row outlives the footage:
    # the keypoints are still readable, the clip is not.
    footage_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # The candidate set, as JSON. A whole-set read and a whole-set write every
    # time, never queried by field, so a column beats a second table here.
    candidates_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"))
    # "ingest" normalises an upload; "analysis" tracks a chosen person. Only
    # the latter has a candidate or an analysis run.
    kind: Mapped[str] = mapped_column(String(16), default="analysis", index=True)
    candidate_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The immutable run this job executes. A job is the mutable record of one
    # attempt; the run is what it was asked to do, fixed at submission.
    analysis_run_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )

    status: Mapped[str] = mapped_column(String(16), index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The status endpoint is polled per video while a job runs.
    __table_args__ = (Index("ix_jobs_video_created", "video_id", "created_at"),)


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    candidate_frame_index: Mapped[int] = mapped_column(Integer)
    selected_candidate_index: Mapped[int] = mapped_column(Integer)
    seed_x: Mapped[float] = mapped_column(Float)
    seed_y: Mapped[float] = mapped_column(Float)
    seed_width: Mapped[float] = mapped_column(Float)
    seed_height: Mapped[float] = mapped_column(Float)
    min_iou: Mapped[float] = mapped_column(Float)
    max_gap_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pose_model: Mapped[str] = mapped_column(String(32))
    max_people: Mapped[int] = mapped_column(Integer)
    # The second-pass settings, snapshotted for the same reason as the rest:
    # a run has to describe what produced it, and these change its output.
    refine_landmarks: Mapped[bool] = mapped_column(Integer, default=1)
    refine_margin: Mapped[float] = mapped_column(Float, default=0.55)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Where this run's keypoints live. Authoritative: one file per run, so two
    # analyses of the same video never overwrite each other.
    keypoints_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
