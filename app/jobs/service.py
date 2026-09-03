"""Submitting analysis jobs.

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
    JobRecord,
    JobRepository,
    JobStatus,
    UnitOfWork,
    VideoRepository,
)
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
    ) -> None:
        self._jobs = jobs
        self._videos = videos
        self._queue = queue
        self._unit_of_work = unit_of_work

    def submit(self, video_id: str, candidate_index: int) -> JobRecord:
        """Queue an analysis of one candidate. Returns immediately."""
        candidates = self._videos.get_candidates(video_id)
        if candidates is None:
            raise UnknownCandidate("no candidates for this video; call /candidates first")
        if candidates.get(candidate_index) is None:
            available = [candidate.index for candidate in candidates.candidates]
            raise UnknownCandidate(
                f"no candidate {candidate_index}; available: {available}"
            )

        job = self._jobs.add(
            JobRecord(
                id=uuid.uuid4().hex,
                video_id=video_id,
                candidate_index=candidate_index,
                status=JobStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
            )
        )
        # Committed before publishing: the worker thread cannot see a row that
        # is still inside this transaction.
        self._unit_of_work.commit()
        self._queue.enqueue(job.id)
        logger.info("queued job %s for video %s candidate %d", job.id, video_id, candidate_index)
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
