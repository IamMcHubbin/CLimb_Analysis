"""SQLAlchemy-backed implementation of the repository interfaces."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import JobRow, VideoRow
from app.db.repository import (
    Candidate,
    CandidateSet,
    JobKind,
    JobRecord,
    JobRepository,
    JobStatus,
    UnitOfWork,
    VideoRecord,
    VideoRepository,
    VideoStatus,
)
from app.geometry import BoundingBox


def _encode_candidates(candidates: CandidateSet) -> str:
    return json.dumps(
        {
            "frame_index": candidates.frame_index,
            "candidates": [
                {
                    "index": candidate.index,
                    "x": candidate.bounding_box.x,
                    "y": candidate.bounding_box.y,
                    "width": candidate.bounding_box.width,
                    "height": candidate.bounding_box.height,
                    "mean_visibility": candidate.mean_visibility,
                }
                for candidate in candidates.candidates
            ],
        }
    )


def _decode_candidates(payload: str) -> CandidateSet:
    data = json.loads(payload)
    return CandidateSet(
        frame_index=int(data["frame_index"]),
        candidates=tuple(
            Candidate(
                index=int(entry["index"]),
                bounding_box=BoundingBox(
                    x=float(entry["x"]),
                    y=float(entry["y"]),
                    width=float(entry["width"]),
                    height=float(entry["height"]),
                ),
                mean_visibility=float(entry["mean_visibility"]),
            )
            for entry in data["candidates"]
        ),
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite has no timezone type; timestamps are written as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _to_record(row: VideoRow) -> VideoRecord:
    return VideoRecord(
        id=row.id,
        original_filename=row.original_filename,
        created_at=_as_utc(row.created_at),
        status=VideoStatus(row.status),
        upload_path=row.upload_path,
        ingest_error=row.ingest_error,
        stored_path=row.stored_path,
        width=row.width,
        height=row.height,
        fps=row.fps,
        frame_count=row.frame_count,
        duration_seconds=row.duration_seconds,
        size_bytes=row.size_bytes,
        source_width=row.source_width,
        source_height=row.source_height,
        source_fps=row.source_fps,
        source_rotation=row.source_rotation,
        source_codec=row.source_codec,
        source_variable_frame_rate=(
            None if row.source_variable_frame_rate is None
            else bool(row.source_variable_frame_rate)
        ),
        keypoints_path=row.keypoints_path,
        analysis_completed_at=_as_utc(row.analysis_completed_at),
        footage_deleted_at=_as_utc(row.footage_deleted_at),
    )


def _to_row(record: VideoRecord) -> VideoRow:
    return VideoRow(
        id=record.id,
        original_filename=record.original_filename,
        created_at=record.created_at,
        status=record.status.value,
        upload_path=record.upload_path,
        ingest_error=record.ingest_error,
        stored_path=record.stored_path,
        width=record.width,
        height=record.height,
        fps=record.fps,
        frame_count=record.frame_count,
        duration_seconds=record.duration_seconds,
        size_bytes=record.size_bytes,
        source_width=record.source_width,
        source_height=record.source_height,
        source_fps=record.source_fps,
        source_rotation=record.source_rotation,
        source_codec=record.source_codec,
        source_variable_frame_rate=(
            None if record.source_variable_frame_rate is None
            else int(record.source_variable_frame_rate)
        ),
        keypoints_path=record.keypoints_path,
        analysis_completed_at=record.analysis_completed_at,
        footage_deleted_at=record.footage_deleted_at,
    )


class SqlAlchemyVideoRepository(VideoRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, video: VideoRecord) -> VideoRecord:
        row = _to_row(video)
        self._session.add(row)
        self._session.flush()
        return _to_record(row)

    def get(self, video_id: str) -> VideoRecord | None:
        row = self._session.get(VideoRow, video_id)
        return _to_record(row) if row is not None else None

    def list(self, limit: int = 50, offset: int = 0) -> list[VideoRecord]:
        stmt = (
            select(VideoRow)
            .order_by(VideoRow.created_at.desc(), VideoRow.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_to_record(row) for row in self._session.scalars(stmt)]

    def set_keypoints_path(self, video_id: str, path: str | None) -> None:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            raise KeyError(video_id)
        row.keypoints_path = path
        row.analysis_completed_at = datetime.now(timezone.utc) if path else None
        self._session.flush()

    def mark_ready(self, video_id: str, normalised: VideoRecord) -> None:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            raise KeyError(video_id)
        row.status = VideoStatus.READY.value
        row.ingest_error = None
        # The raw upload is gone by now; the column must not outlive it.
        row.upload_path = None
        row.width = normalised.width
        row.height = normalised.height
        row.fps = normalised.fps
        row.frame_count = normalised.frame_count
        row.duration_seconds = normalised.duration_seconds
        row.size_bytes = normalised.size_bytes
        row.source_width = normalised.source_width
        row.source_height = normalised.source_height
        row.source_fps = normalised.source_fps
        row.source_rotation = normalised.source_rotation
        row.source_codec = normalised.source_codec
        row.source_variable_frame_rate = int(bool(normalised.source_variable_frame_rate))
        self._session.flush()

    def mark_ingest_failed(self, video_id: str, error: str) -> None:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            raise KeyError(video_id)
        row.status = VideoStatus.FAILED.value
        row.ingest_error = error
        row.upload_path = None
        self._session.flush()

    def mark_footage_deleted(self, video_id: str) -> None:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            raise KeyError(video_id)
        row.footage_deleted_at = datetime.now(timezone.utc)
        self._session.flush()

    def list_footage_to_delete(
        self,
        analysed_before: datetime,
        unanalysed_before: datetime,
    ) -> list[VideoRecord]:
        stmt = select(VideoRow).where(
            VideoRow.footage_deleted_at.is_(None),
            or_(
                VideoRow.analysis_completed_at.isnot(None)
                & (VideoRow.analysis_completed_at < analysed_before),
                VideoRow.analysis_completed_at.is_(None)
                & (VideoRow.created_at < unanalysed_before),
            ),
        )
        return [_to_record(row) for row in self._session.scalars(stmt)]

    def save_candidates(self, video_id: str, candidates: CandidateSet) -> None:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            raise KeyError(video_id)
        row.candidates_json = _encode_candidates(candidates)
        self._session.flush()

    def get_candidates(self, video_id: str) -> CandidateSet | None:
        row = self._session.get(VideoRow, video_id)
        if row is None or not row.candidates_json:
            return None
        return _decode_candidates(row.candidates_json)

    def delete(self, video_id: str) -> bool:
        row = self._session.get(VideoRow, video_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _to_job(row: JobRow) -> JobRecord:
    return JobRecord(
        id=row.id,
        video_id=row.video_id,
        kind=JobKind(row.kind),
        candidate_index=row.candidate_index,
        status=JobStatus(row.status),
        progress=row.progress,
        created_at=_as_utc(row.created_at),
        started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at),
        error=row.error,
    )


class SqlAlchemyJobRepository(JobRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, job: JobRecord) -> JobRecord:
        row = JobRow(
            id=job.id,
            video_id=job.video_id,
            kind=job.kind.value,
            candidate_index=job.candidate_index,
            status=job.status.value,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
        )
        self._session.add(row)
        self._session.flush()
        return _to_job(row)

    def get(self, job_id: str) -> JobRecord | None:
        row = self._session.get(JobRow, job_id)
        return _to_job(row) if row is not None else None

    def list_for_video(self, video_id: str, limit: int = 20) -> list[JobRecord]:
        stmt = (
            select(JobRow)
            .where(JobRow.video_id == video_id)
            .order_by(JobRow.created_at.desc(), JobRow.id.desc())
            .limit(limit)
        )
        return [_to_job(row) for row in self._session.scalars(stmt)]

    def _require(self, job_id: str) -> JobRow:
        row = self._session.get(JobRow, job_id)
        if row is None:
            raise KeyError(job_id)
        return row

    def mark_running(self, job_id: str) -> None:
        row = self._require(job_id)
        row.status = JobStatus.RUNNING.value
        row.started_at = datetime.now(timezone.utc)
        self._session.flush()

    def update_progress(self, job_id: str, progress: float) -> None:
        row = self._require(job_id)
        row.progress = min(1.0, max(0.0, progress))
        self._session.flush()

    def mark_done(self, job_id: str) -> None:
        row = self._require(job_id)
        row.status = JobStatus.DONE.value
        row.progress = 1.0
        row.finished_at = datetime.now(timezone.utc)
        self._session.flush()

    def mark_failed(self, job_id: str, error: str) -> None:
        row = self._require(job_id)
        row.status = JobStatus.FAILED.value
        row.finished_at = datetime.now(timezone.utc)
        row.error = error
        self._session.flush()

    def list_unfinished(self) -> list[JobRecord]:
        stmt = (
            select(JobRow)
            .where(JobRow.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))
            .order_by(JobRow.created_at)
        )
        return [_to_job(row) for row in self._session.scalars(stmt)]


class SqlAlchemyUnitOfWork(UnitOfWork):
    def __init__(self, session: Session) -> None:
        self._session = session

    def commit(self) -> None:
        self._session.commit()
