"""Response models. Separate from the domain records so the wire format can
change without touching persistence."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.db.repository import CandidateSet, VideoRecord


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


class BoxOut(BaseModel):
    """Normalised 0-1 against the frame, so it scales to whatever size the
    client renders the image at."""

    x: float
    y: float
    width: float
    height: float


class CandidateOut(BaseModel):
    index: int
    bounding_box: BoxOut
    mean_visibility: float


class CandidatesOut(BaseModel):
    frame_index: int
    timestamp_seconds: float
    frame_url: str
    frame_width: int
    frame_height: int
    candidates: list[CandidateOut]

    @classmethod
    def from_set(cls, video: VideoRecord, candidate_set: CandidateSet) -> "CandidatesOut":
        return cls(
            frame_index=candidate_set.frame_index,
            timestamp_seconds=candidate_set.frame_index / video.fps if video.fps else 0.0,
            # A URL rather than an inlined image: the browser can render it
            # with <img> and cache it, and the JSON stays small.
            frame_url=f"/videos/{video.id}/candidates/frame.jpg",
            frame_width=video.width,
            frame_height=video.height,
            candidates=[
                CandidateOut(
                    index=candidate.index,
                    bounding_box=BoxOut(
                        x=candidate.bounding_box.x,
                        y=candidate.bounding_box.y,
                        width=candidate.bounding_box.width,
                        height=candidate.bounding_box.height,
                    ),
                    mean_visibility=candidate.mean_visibility,
                )
                for candidate in candidate_set.candidates
            ],
        )
