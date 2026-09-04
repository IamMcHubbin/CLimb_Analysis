"""Response models. Separate from the domain records so the wire format can
change without touching persistence."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.db.repository import (
    AnalysisRun,
    CandidateSet,
    JobKind,
    JobRecord,
    JobStatus,
    VideoRecord,
    VideoStatus,
)


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
    # "pending" until normalisation finishes; the fields below are null until
    # then, and "failed" carries the reason in ingest_error.
    status: VideoStatus
    ingest_error: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    has_keypoints: bool
    # False once the clip has been deleted. The keypoints outlive it, so the
    # row is still useful - there is just nothing left to play.
    has_footage: bool
    footage_expires_at: datetime | None
    source: SourceInfo | None = None

    @classmethod
    def from_record(
        cls, record: VideoRecord, footage_expires_at: datetime | None = None
    ) -> "VideoOut":
        source = None
        if record.source_codec is not None:
            source = SourceInfo(
                width=record.source_width,
                height=record.source_height,
                fps=record.source_fps,
                rotation=record.source_rotation,
                codec=record.source_codec,
                variable_frame_rate=bool(record.source_variable_frame_rate),
            )
        return cls(
            id=record.id,
            original_filename=record.original_filename,
            created_at=record.created_at,
            status=record.status,
            ingest_error=record.ingest_error,
            width=record.width,
            height=record.height,
            fps=record.fps,
            frame_count=record.frame_count,
            duration_seconds=record.duration_seconds,
            size_bytes=record.size_bytes,
            has_keypoints=record.keypoints_path is not None,
            has_footage=record.has_footage,
            footage_expires_at=footage_expires_at,
            source=source,
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


class AnalyseRequest(BaseModel):
    """Which of the detected people to follow."""

    candidate_index: int


class JobOut(BaseModel):
    id: str
    video_id: str
    kind: JobKind
    # Analysis jobs only: the candidate picked, and the immutable run that
    # records what was actually asked for.
    candidate_index: int | None = None
    analysis_run_id: str | None = None
    status: JobStatus
    # 0-1 over the frames processed so far.
    progress: float
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @classmethod
    def from_record(cls, job: JobRecord) -> "JobOut":
        return cls(
            id=job.id,
            video_id=job.video_id,
            kind=job.kind,
            candidate_index=job.candidate_index,
            analysis_run_id=job.analysis_run_id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
        )


class AnalysisExecutionOut(BaseModel):
    """Mutable job state associated with an immutable analysis run."""

    job_id: str
    status: JobStatus
    progress: float
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @classmethod
    def from_record(cls, job: JobRecord) -> "AnalysisExecutionOut":
        return cls(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
        )


class PoseConfigurationOut(BaseModel):
    model: str
    max_people: int


class TrackingConfigurationOut(BaseModel):
    min_iou: float
    max_gap_frames: int | None


class AnalysisRunSummaryOut(BaseModel):
    id: str
    video_id: str
    created_at: datetime
    selected_candidate_index: int
    pose_model: str
    result_available: bool
    execution: AnalysisExecutionOut | None

    @classmethod
    def from_records(
        cls, run: AnalysisRun, job: JobRecord | None
    ) -> "AnalysisRunSummaryOut":
        return cls(
            id=run.id,
            video_id=run.video_id,
            created_at=run.created_at,
            selected_candidate_index=run.selected_candidate_index,
            pose_model=run.pose_model,
            result_available=run.keypoints_path is not None,
            execution=AnalysisExecutionOut.from_record(job) if job is not None else None,
        )


class AnalysisRunDetailOut(BaseModel):
    id: str
    video_id: str
    created_at: datetime
    seed_frame_index: int
    selected_candidate_index: int
    seed_bounding_box: BoxOut
    pose_configuration: PoseConfigurationOut
    tracking_configuration: TrackingConfigurationOut
    result_available: bool
    keypoints_url: str | None
    execution: AnalysisExecutionOut | None

    @classmethod
    def from_records(
        cls, run: AnalysisRun, job: JobRecord | None
    ) -> "AnalysisRunDetailOut":
        return cls(
            id=run.id,
            video_id=run.video_id,
            created_at=run.created_at,
            seed_frame_index=run.candidate_frame_index,
            selected_candidate_index=run.selected_candidate_index,
            seed_bounding_box=BoxOut(
                x=run.seed_box.x,
                y=run.seed_box.y,
                width=run.seed_box.width,
                height=run.seed_box.height,
            ),
            pose_configuration=PoseConfigurationOut(
                model=run.pose_model,
                max_people=run.max_people,
            ),
            tracking_configuration=TrackingConfigurationOut(
                min_iou=run.min_iou,
                max_gap_frames=run.max_gap_frames,
            ),
            result_available=run.keypoints_path is not None,
            keypoints_url=(
                f"/videos/{run.video_id}/keypoints?analysis_run_id={run.id}"
                if run.keypoints_path is not None
                else None
            ),
            execution=AnalysisExecutionOut.from_record(job) if job is not None else None,
        )


class KeypointsOut(BaseModel):
    """A finished track, laid out for drawing.

    ``frames`` is index-aligned with the video: entry N is frame N, and a null
    entry is a gap the tracker could not fill. Each present entry is a list of
    [x, y, visibility] per landmark, normalised 0-1 against the frame.
    """

    video_id: str
    analysis_run_id: str | None = None
    fps: float
    frame_count: int
    landmark_names: list[str]
    landmark_connections: list[list[int]]
    pose_model: str
    min_iou: float
    tracked_frame_count: int
    gap_frame_count: int
    frames: list[list[list[float]] | None]
    # The tracker's IoU against the previous frame, per frame; null on gaps.
    # Exposed because how confident each match was is the question the whole
    # proof of concept is asking.
    match_iou: list[float | None]
