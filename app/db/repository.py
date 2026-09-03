"""Domain records and the repository interface.

Routes, services and workers depend on these abstractions only. Swapping SQLite
for something else means writing a new implementation of ``VideoRepository``
and changing one line of wiring.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, replace
from datetime import datetime

from app.geometry import BoundingBox


@dataclass(frozen=True)
class VideoRecord:
    """A normalised video and the provenance of the upload it came from."""

    id: str
    original_filename: str
    created_at: datetime

    stored_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    size_bytes: int

    source_width: int
    source_height: int
    source_fps: float
    source_rotation: int
    source_codec: str
    source_variable_frame_rate: bool

    keypoints_path: str | None = None

    def with_keypoints_path(self, path: str | None) -> "VideoRecord":
        return replace(self, keypoints_path=path)


@dataclass(frozen=True)
class Candidate:
    """One person detected in the candidate frame, offered to the user to pick.

    ``index`` is the handle the client sends back to start an analysis, so it
    has to stay stable for as long as the set is stored.
    """

    index: int
    bounding_box: BoundingBox
    mean_visibility: float


@dataclass(frozen=True)
class CandidateSet:
    """The people found in one frame of a video.

    Stored rather than recomputed because the index the user picks only means
    something against the exact detection run that produced it, and because
    tracking is seeded from the chosen box.
    """

    frame_index: int
    candidates: tuple[Candidate, ...]

    def get(self, index: int) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.index == index:
                return candidate
        return None


class VideoRepository(abc.ABC):
    """Persistence for video metadata."""

    @abc.abstractmethod
    def add(self, video: VideoRecord) -> VideoRecord:
        """Store a new video. Raises on duplicate id."""

    @abc.abstractmethod
    def get(self, video_id: str) -> VideoRecord | None:
        """Return the video, or None if it does not exist."""

    @abc.abstractmethod
    def list(self, limit: int = 50, offset: int = 0) -> list[VideoRecord]:
        """Most recently uploaded first."""

    @abc.abstractmethod
    def set_keypoints_path(self, video_id: str, path: str | None) -> None:
        """Point the video at its finished keypoint file."""

    @abc.abstractmethod
    def save_candidates(self, video_id: str, candidates: CandidateSet) -> None:
        """Replace the stored candidate set for a video."""

    @abc.abstractmethod
    def get_candidates(self, video_id: str) -> CandidateSet | None:
        """The stored candidate set, or None if detection has not run."""

    @abc.abstractmethod
    def delete(self, video_id: str) -> bool:
        """Remove the row. Returns False if it was not there."""
