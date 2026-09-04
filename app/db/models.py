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

    # The normalised video: constant fps, rotation baked in, long edge capped.
    # Path is relative to the data directory. The original is never kept.
    stored_path: Mapped[str] = mapped_column(String(512))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    frame_count: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float] = mapped_column(Float)
    size_bytes: Mapped[int] = mapped_column(Integer)

    # What the upload looked like before normalisation. Kept for debugging
    # ingest problems - orientation and frame timing bugs are invisible
    # afterwards, so the evidence has to be recorded here.
    source_width: Mapped[int] = mapped_column(Integer)
    source_height: Mapped[int] = mapped_column(Integer)
    source_fps: Mapped[float] = mapped_column(Float)
    source_rotation: Mapped[int] = mapped_column(Integer)
    source_codec: Mapped[str] = mapped_column(String(64))
    source_variable_frame_rate: Mapped[bool] = mapped_column(Integer)

    # Set once a pose analysis job finishes. Relative to the data directory.
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
    candidate_index: Mapped[int] = mapped_column(Integer)

    status: Mapped[str] = mapped_column(String(16), index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The status endpoint is polled per video while a job runs.
    __table_args__ = (Index("ix_jobs_video_created", "video_id", "created_at"),)
