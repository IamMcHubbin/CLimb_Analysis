"""The analysis job: pose over every frame, tracked to one person, stored.

Seeding is the awkward part. The user picks their climber in a frame in the
middle of the clip, but the track has to cover the whole clip, and where that
person was at frame 0 is not known. So the clip is decoded once, forwards;
frames from the seed onward are tracked as they are decoded, and the
detections before the seed are buffered and tracked backwards afterwards. Every
frame gets exactly one pose inference, and the buffer holds only the first part
of the clip - which is what bounds how long a video this can handle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from app.config import Settings, settings as default_settings
from app.geometry import BoundingBox
from app.db.repository import AnalysisRunRepository, JobRepository, VideoRecord, VideoRepository
from app.db.session import session_scope
from app.db.sqlalchemy_repository import SqlAlchemyAnalysisRunRepository, SqlAlchemyJobRepository, SqlAlchemyVideoRepository
from app.frames import FrameReader, timestamp_ms_for_frame
from app.keypoints import KeypointMetadata, KeypointStore, ParquetKeypointStore
from app.pose import PoseEstimator, RunningMode, create_pose_estimator
from app.pose.refine import refine_landmarks
from app.retention import FootageRetention
from app.pose.base import PersonPose
from app.tracking import IouTracker, TrackedFrame, TrackingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackResult:
    """A finished track plus the schema of the model that produced it."""

    frames: list[TrackedFrame]
    landmark_names: tuple[str, ...]
    landmark_connections: tuple[tuple[int, int], ...]


VideoEstimatorFactory = Callable[[], AbstractContextManager[PoseEstimator]]


class AnalysisFailed(RuntimeError):
    pass


class AnalysisJobHandler:
    """Runs one analysis job to completion.

    Called on the worker thread, so it opens its own short-lived sessions
    rather than holding one across the whole run: a job takes minutes, and a
    transaction held that long would block every status request behind it.
    """

    def __init__(
        self,
        settings: Settings = default_settings,
        estimator_factory: VideoEstimatorFactory | None = None,
        keypoint_store: KeypointStore | None = None,
        tracking_config: TrackingConfig | None = None,
        retention: FootageRetention | None = None,
    ) -> None:
        self._settings = settings
        self._estimator_factory = estimator_factory
        self._store = keypoint_store or ParquetKeypointStore(settings)
        self._tracking_config = tracking_config or TrackingConfig()
        self._retention = retention or FootageRetention(settings)

    def _default_estimator(
        self, variant: str | None = None, max_people: int | None = None
    ) -> AbstractContextManager[PoseEstimator]:
        # VIDEO mode: the model may use temporal context between frames.
        return create_pose_estimator(
            mode=RunningMode.VIDEO,
            variant=variant or self._settings.pose_model,
            num_poses=max_people if max_people is not None else self._settings.max_people,
            settings=self._settings,
        )

    def __call__(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except Exception as exc:
            logger.exception("analysis job %s failed", job_id)
            try:
                with session_scope() as session:
                    SqlAlchemyJobRepository(session).mark_failed(job_id, str(exc))
            except Exception:
                # The row is gone, or the database is unreachable. Nothing more
                # can be recorded, and raising here would only mask the real
                # failure above.
                logger.exception("could not record the failure of job %s", job_id)

    def _run(self, job_id: str) -> None:
        with session_scope() as session:
            jobs: JobRepository = SqlAlchemyJobRepository(session)
            videos: VideoRepository = SqlAlchemyVideoRepository(session)
            runs: AnalysisRunRepository = SqlAlchemyAnalysisRunRepository(session)
            job = jobs.get(job_id)
            if job is None:
                raise AnalysisFailed(f"job {job_id} no longer exists")
            video = videos.get(job.video_id)
            if video is None:
                raise AnalysisFailed(f"video {job.video_id} no longer exists")
            if job.analysis_run_id is None:
                raise AnalysisFailed("job has no immutable analysis run")
            run = runs.get(job.analysis_run_id)
            if run is None or run.video_id != video.id:
                raise AnalysisFailed("analysis run is missing or belongs to another video")
            jobs.mark_running(job_id)

        # Everything below reads the run, never the live settings: the model,
        # the thresholds and the refinement options are whatever they were when
        # this analysis was asked for. Changing the global model must not
        # retroactively change what a queued job does.
        tracking_config = TrackingConfig(run.min_iou, run.max_gap_frames)
        estimator_factory = self._estimator_factory or (
            lambda: self._default_estimator(run.pose_model, run.max_people)
        )
        result = self._track(
            job_id, video, run.candidate_frame_index, run.seed_box,
            tracking_config, estimator_factory, run.refine_landmarks,
        )
        if run.refine_landmarks:
            # The same factory, so the second pass uses the run's model too.
            result = self._refine(job_id, video, result, estimator_factory, run.refine_margin)

        metadata = KeypointMetadata(
            video_id=video.id,
            fps=video.fps,
            frame_count=video.frame_count,
            landmark_names=result.landmark_names,
            landmark_connections=result.landmark_connections,
            pose_model=run.pose_model,
            min_iou=tracking_config.min_iou,
            analysis_run_id=run.id,
        )
        path = self._store.write(metadata, result.frames)

        with session_scope() as session:
            videos = SqlAlchemyVideoRepository(session)
            # The run's path is the authoritative one; the video's is a
            # convenience pointer at whatever finished most recently.
            SqlAlchemyAnalysisRunRepository(session).set_keypoints_path(run.id, path)
            videos.set_latest_keypoints_path(video.id, path)
            SqlAlchemyJobRepository(session).mark_done(job_id)
            # Swept here as well as on the timer, so a zero retention window
            # means the footage is gone the moment the job finishes rather
            # than up to a sweep interval later.
            self._retention.sweep(videos)

        gaps = sum(1 for frame in result.frames if frame.is_gap)
        logger.info(
            "job %s finished: %d/%d frames tracked (%d gaps)",
            job_id,
            len(result.frames) - gaps,
            len(result.frames),
            gaps,
        )

    def _track(
        self,
        job_id: str,
        video: VideoRecord,
        seed_index: int,
        seed_box: BoundingBox,
        tracking_config: TrackingConfig,
        estimator_factory: VideoEstimatorFactory,
        refine_pass_follows: bool,
    ) -> TrackResult:
        video_path = self._settings.resolve(video.stored_path)

        forward = IouTracker(seed_box, tracking_config)
        results: dict[int, TrackedFrame] = {}
        before_seed: dict[int, tuple[PersonPose, ...]] = {}

        # Write progress at most every percent; SQLite is not the place for a
        # write per frame.
        update_every = max(1, video.frame_count // 100)

        with estimator_factory() as estimator, FrameReader(video_path) as reader:
            landmark_names = estimator.landmark_names
            landmark_connections = estimator.landmark_connections

            for frame_index, frame in reader:
                if frame_index >= video.frame_count:
                    break
                people = estimator.detect(
                    frame, timestamp_ms_for_frame(frame_index, video.fps)
                )
                if frame_index < seed_index:
                    # Held for the backward pass; tracking these needs an
                    # anchor that does not exist until the seed frame.
                    before_seed[frame_index] = people
                else:
                    results[frame_index] = forward.update(frame_index, people)

                if frame_index % update_every == 0:
                    share = 0.5 if refine_pass_follows else 1.0
                    self._report_progress(
                        job_id, share * frame_index / max(1, video.frame_count)
                    )

        # Backward from the seed, so the start of the clip is covered too.
        backward = IouTracker(seed_box, tracking_config)
        for frame_index in range(seed_index - 1, -1, -1):
            people = before_seed.get(frame_index, ())
            results[frame_index] = backward.update(frame_index, people)

        return TrackResult(
            frames=[
                results.get(index, TrackedFrame(index, None))
                for index in range(video.frame_count)
            ],
            landmark_names=landmark_names,
            landmark_connections=landmark_connections,
        )

    def _refine(
        self,
        job_id: str,
        video: VideoRecord,
        result: TrackResult,
        estimator_factory: VideoEstimatorFactory,
        margin: float,
    ) -> TrackResult:
        """Second pass: better landmarks from a crop around the tracked person.

        Takes the run's estimator factory and margin, not the current global
        settings, so both passes of one analysis use the same model.

        A failure here is not fatal - the coarse track is still a usable
        answer, and losing the whole job over an improvement would be a poor
        trade.
        """
        video_path = self._settings.resolve(video.stored_path)
        try:
            with estimator_factory() as estimator, FrameReader(video_path) as reader:
                frames = refine_landmarks(
                    result.frames,
                    reader,
                    estimator,
                    margin=margin,
                    # The first pass covered 0-50% of the bar; this covers the rest.
                    on_progress=lambda f: self._report_progress(job_id, 0.5 + f * 0.5),
                )
        except Exception:
            logger.exception("landmark refinement failed; keeping the first-pass track")
            return result
        return TrackResult(
            frames=frames,
            landmark_names=result.landmark_names,
            landmark_connections=result.landmark_connections,
        )

    def _report_progress(self, job_id: str, fraction: float) -> None:
        with session_scope() as session:
            SqlAlchemyJobRepository(session).update_progress(job_id, fraction)
