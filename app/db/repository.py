"""Domain records and the repository interface.

Routes, services and workers depend on these abstractions only. Swapping SQLite
for something else means writing a new implementation of ``VideoRepository``
and changing one line of wiring.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from app.geometry import BoundingBox


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


@dataclass(frozen=True)
class JobRecord:
    """One analysis run over one video.

    ``progress`` is 0-1 over the frames processed so far.
    """

    id: str
    video_id: str
    candidate_index: int
    status: JobStatus
    progress: float
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


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
    analysis_completed_at: datetime | None = None
    footage_deleted_at: datetime | None = None

    def with_keypoints_path(self, path: str | None) -> "VideoRecord":
        return replace(self, keypoints_path=path)

    @property
    def has_footage(self) -> bool:
        return self.footage_deleted_at is None

    @property
    def is_analysed(self) -> bool:
        return self.keypoints_path is not None


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
        """Point the video at its finished keypoint file, and stamp the time.

        The timestamp is what the retention sweep counts from.
        """

    @abc.abstractmethod
    def mark_footage_deleted(self, video_id: str) -> None:
        """Record that the video file is gone. The row and keypoints remain."""

    @abc.abstractmethod
    def list_footage_to_delete(
        self,
        analysed_before: datetime,
        unanalysed_before: datetime,
    ) -> list["VideoRecord"]:
        """Videos still holding footage that is past its retention window.

        Two windows, because the two cases are different: an analysed video has
        served its purpose and is counted from when analysis finished, while an
        upload nobody ever analysed is abandoned and counted from when it
        arrived.
        """

    @abc.abstractmethod
    def save_candidates(self, video_id: str, candidates: CandidateSet) -> None:
        """Replace the stored candidate set for a video."""

    @abc.abstractmethod
    def get_candidates(self, video_id: str) -> CandidateSet | None:
        """The stored candidate set, or None if detection has not run."""

    @abc.abstractmethod
    def delete(self, video_id: str) -> bool:
        """Remove the row. Returns False if it was not there."""


class UnitOfWork(abc.ABC):
    """Explicit control over when written rows become visible elsewhere.

    Needed because the job queue runs on another thread: an id handed to the
    worker before its row is committed refers, from that thread, to a job that
    does not exist.
    """

    @abc.abstractmethod
    def commit(self) -> None:
        """Make everything written so far durable and visible."""


class JobRepository(abc.ABC):
    """Persistence for job state.

    Job state is in the database rather than in the queue so that a status
    request answers from durable storage - a page refresh, or a restart, sees
    what actually happened.
    """

    @abc.abstractmethod
    def add(self, job: JobRecord) -> JobRecord:
        """Store a newly queued job."""

    @abc.abstractmethod
    def get(self, job_id: str) -> JobRecord | None:
        """Return the job, or None if it does not exist."""

    @abc.abstractmethod
    def list_for_video(self, video_id: str, limit: int = 20) -> list[JobRecord]:
        """Jobs for one video, newest first."""

    @abc.abstractmethod
    def mark_running(self, job_id: str) -> None:
        """Move a job to running and stamp its start time."""

    @abc.abstractmethod
    def update_progress(self, job_id: str, progress: float) -> None:
        """Record fractional progress, 0-1."""

    @abc.abstractmethod
    def mark_done(self, job_id: str) -> None:
        """Move a job to done and stamp its finish time."""

    @abc.abstractmethod
    def mark_failed(self, job_id: str, error: str) -> None:
        """Move a job to failed, recording why."""

    @abc.abstractmethod
    def list_unfinished(self) -> list[JobRecord]:
        """Jobs left queued or running, for recovery after a restart."""
