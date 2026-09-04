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


class VideoStatus(str, Enum):
    """Where an upload has got to.

    A video is PENDING from the moment its bytes land until ffmpeg has
    normalised it. Only a READY video has metadata; see ``require_ready``.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class JobKind(str, Enum):
    INGEST = "ingest"
    ANALYSIS = "analysis"


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
    kind: JobKind
    # Only analysis jobs have one.
    candidate_index: int | None
    status: JobStatus
    progress: float
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    analysis_run_id: str | None = None


@dataclass(frozen=True)
class AnalysisRun:
    """Immutable snapshot of one requested analysis.

    Everything that decides what the worker produces is fixed here at
    submission: which person, in which frame, with which model and thresholds.
    The candidate set on the video is mutable - re-running the picker replaces
    it - so a worker reading it at execution time could analyse a different
    person than the one that was chosen. This is what it reads instead.

    It also makes a result self-describing. Two runs over one video with
    different models are comparable because each carries the settings that
    produced it.
    """

    id: str
    video_id: str
    candidate_frame_index: int
    selected_candidate_index: int
    seed_box: BoundingBox
    min_iou: float
    max_gap_frames: int | None
    pose_model: str
    max_people: int
    created_at: datetime
    # Second-pass settings, snapshotted for the same reason as the rest.
    refine_landmarks: bool = True
    refine_margin: float = 0.55
    # The authoritative location of this run's keypoints. One file per run.
    keypoints_path: str | None = None


@dataclass(frozen=True)
class VideoRecord:
    """An upload, and once it has been normalised, the video it became.

    Everything below ``status`` is None until normalisation finishes, because
    none of it is known before ffmpeg has run: not the frame count, not the
    real dimensions, not even whether the file was a video at all. Code that
    needs those values should go through ``require_ready`` rather than
    assuming.
    """

    id: str
    original_filename: str
    created_at: datetime
    status: VideoStatus

    stored_path: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None

    source_width: int | None = None
    source_height: int | None = None
    source_fps: float | None = None
    source_rotation: int | None = None
    source_codec: str | None = None
    source_variable_frame_rate: bool | None = None

    upload_path: str | None = None
    ingest_error: str | None = None

    # The most recently completed run's artifact, not the canonical one; see
    # set_latest_keypoints_path and AnalysisRun.keypoints_path.
    keypoints_path: str | None = None
    analysis_completed_at: datetime | None = None
    footage_deleted_at: datetime | None = None

    def with_keypoints_path(self, path: str | None) -> "VideoRecord":
        return replace(self, keypoints_path=path)

    @property
    def is_ready(self) -> bool:
        return self.status is VideoStatus.READY

    def require_ready(self) -> "VideoRecord":
        """Return self, or raise if the video has no metadata yet."""
        if not self.is_ready:
            raise VideoNotReady(self.id, self.status)
        return self

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
    def set_latest_keypoints_path(self, video_id: str, path: str | None) -> None:
        """Record the most recently completed run's artifact for this video.

        A convenience for callers that just want "the latest result". The
        authoritative per-analysis location is AnalysisRun.keypoints_path;
        this column exists so a video can answer "has anything finished?"
        without a join, and so older clients keep working.

        Also stamps the completion time, which is what retention counts from.
        """

    @abc.abstractmethod
    def mark_ready(self, video_id: str, normalised: "VideoRecord") -> None:
        """Fill in the metadata a finished normalisation produced."""

    @abc.abstractmethod
    def mark_ingest_failed(self, video_id: str, error: str) -> None:
        """Record that this upload could not be normalised."""

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


class VideoNotReady(RuntimeError):
    """Raised when a video's metadata is used before normalisation finished."""

    def __init__(self, video_id: str, status: "VideoStatus") -> None:
        super().__init__(f"video {video_id} is {status.value}, not ready")
        self.video_id = video_id
        self.status = status


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


class AnalysisRunRepository(abc.ABC):
    """Persistence for immutable analysis runs."""

    @abc.abstractmethod
    def add(self, run: AnalysisRun) -> AnalysisRun:
        """Store a newly submitted run."""

    @abc.abstractmethod
    def get(self, run_id: str) -> AnalysisRun | None:
        """Return the run, or None if it does not exist."""

    @abc.abstractmethod
    def list_for_video(self, video_id: str, limit: int = 20) -> list[AnalysisRun]:
        """Runs for one video, newest first."""

    @abc.abstractmethod
    def set_keypoints_path(self, run_id: str, path: str) -> None:
        """Record where this run's finished keypoints were written."""

    @abc.abstractmethod
    def referenced_keypoint_paths(self) -> set[str]:
        """Every keypoint path any run points at.

        The retention sweep needs this: an artifact is only an orphan if no
        run claims it, and deleting a referenced one would silently destroy a
        result somebody can still ask for.
        """
