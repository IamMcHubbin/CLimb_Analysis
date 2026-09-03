"""Response models. Separate from the domain records so the wire format can
change without touching persistence."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.db.repository import VideoRecord


class SourceInfo(BaseModel):
    """What the upload looked like before normalisation."""

    width: int
    height: int
    fps: float
    rotation: int
    codec: str
    variable_frame_rate: bool


class VideoOut(BaseModel):
    id: str
    original_filename: str
    created_at: datetime
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    size_bytes: int
    has_keypoints: bool
    source: SourceInfo

    @classmethod
    def from_record(cls, record: VideoRecord) -> "VideoOut":
        return cls(
            id=record.id,
            original_filename=record.original_filename,
            created_at=record.created_at,
            width=record.width,
            height=record.height,
            fps=record.fps,
            frame_count=record.frame_count,
            duration_seconds=record.duration_seconds,
            size_bytes=record.size_bytes,
            has_keypoints=record.keypoints_path is not None,
            source=SourceInfo(
                width=record.source_width,
                height=record.source_height,
                fps=record.source_fps,
                rotation=record.source_rotation,
                codec=record.source_codec,
                variable_frame_rate=record.source_variable_frame_rate,
            ),
        )
