"""Submitting analysis jobs.

Submitting an analysis writes two rows. The AnalysisRun is the immutable
record of what was asked for - which person, in which frame, with which model
and thresholds - and the job is the mutable record of one attempt to do it.
Everything that decides the output is snapshotted onto the run at this point,
so nothing the user changes afterwards can alter a queued analysis.

Ordering matters and is easy to get wrong: the job row must be *committed*,
not merely written, before its id goes on the queue. The worker runs on
another thread with its own session, so an id published inside an open
transaction refers to a job that thread cannot see, and the job fails
immediately with "no longer exists".
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.db.repository import (
    AnalysisRun,
    AnalysisRunRepository,
    JobKind,
    JobRecord,
    JobRepository,
    JobStatus,
    UnitOfWork,
    VideoRepository,
)
from app.tracking import TrackingConfig
from app.config import Settings, settings as default_settings
from app.jobs.base import JobQueue

logger = logging.getLogger(__name__)


class UnknownCandidate(ValueError):
    pass


class JobService:
    def __init__(
        self,
        jobs: JobRepository,
        videos: VideoRepository,
        queue: JobQueue,
        unit_of_work: UnitOfWork,
        analysis_runs: AnalysisRunRepository,
        settings: Settings = default_settings,
        tracking_config: TrackingConfig | None = None,
    ) -> None:
        self._jobs = jobs
        self._videos = videos
        self._queue = queue
        self._unit_of_work = unit_of_work
        self._analysis_runs = analysis_runs
        self._settings = settings
        self._tracking_config = tracking_config or TrackingConfig()

    def submit_ingest(self, video_id: str) -> JobRecord:
        """Queue normalisation of a freshly uploaded video."""
        job = self._jobs.add(
            JobRecord(
                id=uuid.uuid4().hex,
                video_id=video_id,
                kind=JobKind.INGEST,
                candidate_index=None,
                status=JobStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
            )
        )
        self._unit_of_work.commit()
        self._queue.enqueue(job.id)
        logger.info("queued ingest job %s for video %s", job.id, video_id)
        return job

    def submit(self, video_id: str, candidate_index: int) -> JobRecord:
        """Queue an analysis of one candidate. Returns immediately."""
        candidates = self._videos.get_candidates(video_id)
        if candidates is None:
            raise UnknownCandidate("no candidates for this video; call /candidates first")
        candidate = candidates.get(candidate_index)
        if candidate is None:
            available = [entry.index for entry in candidates.candidates]
            raise UnknownCandidate(
                f"no candidate {candidate_index}; available: {available}"
            )

        # The chosen box is copied into the run here, not looked up later. The
        # candidate set is mutable - re-running the picker replaces it - so a
        # worker that read it at execution time could analyse a different
        # person than the one that was picked.
        run = self._analysis_runs.add(
            AnalysisRun(
                id=uuid.uuid4().hex,
                video_id=video_id,
                candidate_frame_index=candidates.frame_index,
                selected_candidate_index=candidate_index,
                seed_box=candidate.bounding_box,
                min_iou=self._tracking_config.min_iou,
                max_gap_frames=self._tracking_config.max_gap_frames,
                pose_model=self._settings.pose_model,
                max_people=self._settings.max_people,
                refine_landmarks=self._settings.refine_landmarks,
                refine_margin=self._settings.refine_margin,
                created_at=datetime.now(timezone.utc),
            )
        )
        job = self._jobs.add(
            JobRecord(
                id=uuid.uuid4().hex,
                video_id=video_id,
                kind=JobKind.ANALYSIS,
                candidate_index=candidate_index,
                status=JobStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
                analysis_run_id=run.id,
            )
        )
        # Committed before publishing: the worker thread cannot see a row that
        # is still inside this transaction.
        self._unit_of_work.commit()
        self._queue.enqueue(job.id)
        logger.info(
            "queued job %s (run %s) for video %s candidate %d",
            job.id, run.id, video_id, candidate_index,
        )
        return job


def recover_unfinished_jobs(jobs: JobRepository, queue: JobQueue) -> None:
    """Deal with jobs left behind by a restart.

    Anything still marked running was interrupted mid-flight - the process that
    owned it is gone, so it is failed rather than left to sit at "running"
    forever. Anything still queued never started, so it is put back on the
    queue.
    """
    for job in jobs.list_unfinished():
        if job.status is JobStatus.RUNNING:
            jobs.mark_failed(job.id, "interrupted by a server restart")
            logger.warning("failed job %s: it was running when the server stopped", job.id)
        else:
            queue.enqueue(job.id)
            logger.info("re-queued job %s after restart", job.id)
