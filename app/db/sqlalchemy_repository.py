"""SQLAlchemy-backed implementation of the repository interfaces."""

from __future__ import annotations

import json
from datetime import timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import VideoRow
from app.db.repository import Candidate, CandidateSet, VideoRecord, VideoRepository
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


def _to_record(row: VideoRow) -> VideoRecord:
    # SQLite has no timezone type, so timestamps come back naive. They are
    # written as UTC, so that is what they are read back as.
    created_at = row.created_at
    if created_at is not None and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return VideoRecord(
        id=row.id,
        original_filename=row.original_filename,
        created_at=created_at,
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
        source_variable_frame_rate=bool(row.source_variable_frame_rate),
        keypoints_path=row.keypoints_path,
    )


def _to_row(record: VideoRecord) -> VideoRow:
    return VideoRow(
        id=record.id,
        original_filename=record.original_filename,
        created_at=record.created_at,
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
        source_variable_frame_rate=int(record.source_variable_frame_rate),
        keypoints_path=record.keypoints_path,
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
        self._session.flush()

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
