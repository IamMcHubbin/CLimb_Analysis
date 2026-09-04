"""The queue, and the ordering rules around it."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from app.db import SqlAlchemyJobRepository, SqlAlchemyVideoRepository, session_scope
from app.db.repository import JobKind, JobRecord, JobStatus
from app.jobs.base import JobQueue
from app.jobs.inprocess import ThreadedJobQueue
from app.jobs.service import JobService, UnknownCandidate, recover_unfinished_jobs


class RecordingQueue(JobQueue):
    """Captures what was enqueued, without running anything."""

    def __init__(self):
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)

    def start(self) -> None:
        pass

    def stop(self, timeout: float | None = None) -> None:
        pass

    @property
    def depth(self) -> int:
        return len(self.enqueued)


# --------------------------------------------------------------- the thread


def test_runs_jobs_in_order_one_at_a_time():
    seen: list[str] = []
    concurrent = []
    running = threading.Lock()

    def handler(job_id: str) -> None:
        acquired = running.acquire(blocking=False)
        concurrent.append(acquired)
        time.sleep(0.01)
        seen.append(job_id)
        if acquired:
            running.release()

    queue = ThreadedJobQueue(handler)
    queue.start()
    for index in range(5):
        queue.enqueue(f"job{index}")
    queue.stop(timeout=10)

    assert seen == [f"job{index}" for index in range(5)]
    # Never two at once: every job got the lock uncontended.
    assert all(concurrent)


def test_a_failing_handler_does_not_kill_the_worker():
    seen: list[str] = []

    def handler(job_id: str) -> None:
        if job_id == "boom":
            raise RuntimeError("handler exploded")
        seen.append(job_id)

    queue = ThreadedJobQueue(handler)
    queue.start()
    queue.enqueue("boom")
    queue.enqueue("after")
    queue.stop(timeout=10)

    # The job after the failure still ran; one bad job must not stop the queue.
    assert seen == ["after"]


def test_start_is_idempotent():
    queue = ThreadedJobQueue(lambda job_id: None)
    queue.start()
    queue.start()
    queue.stop(timeout=10)


def test_stop_without_start_is_harmless():
    ThreadedJobQueue(lambda job_id: None).stop(timeout=1)


# -------------------------------------------------------------- the service


@pytest.fixture
def video_with_candidates(settings, ingest_video):
    from contextlib import contextmanager

    from app.candidates import CandidateService

    from tests.test_candidates import FakeEstimator, _person

    video = ingest_video(seconds=1.0)
    with session_scope() as session:
        @contextmanager
        def factory():
            yield FakeEstimator([_person(0.1, 0.1, 0.3)])

        CandidateService(
            SqlAlchemyVideoRepository(session), settings, estimator_factory=factory
        ).detect(video)
    return video


def test_submitting_queues_a_persisted_job(settings, video_with_candidates):
    queue = RecordingQueue()
    with session_scope() as session:
        jobs = SqlAlchemyJobRepository(session)
        service = JobService(
            jobs,
            SqlAlchemyVideoRepository(session),
            queue,
            _CommitTracker(session, queue),
        )
        job = service.submit(video_with_candidates.id, 0)

    assert job.status is JobStatus.QUEUED
    assert queue.enqueued == [job.id]
    # And the row is readable from a different session, which is what the
    # worker thread will be doing.
    with session_scope() as session:
        assert SqlAlchemyJobRepository(session).get(job.id) is not None


class _CommitTracker:
    """A unit of work that records whether commit happened before enqueueing."""

    def __init__(self, session, queue: RecordingQueue):
        self._session = session
        self._queue = queue
        self.enqueued_at_commit: list[str] = []

    def commit(self) -> None:
        self.enqueued_at_commit = list(self._queue.enqueued)
        self._session.commit()


def test_the_job_is_committed_before_it_is_enqueued(settings, video_with_candidates):
    """The worker runs on another thread and cannot see an open transaction."""
    queue = RecordingQueue()
    with session_scope() as session:
        tracker = _CommitTracker(session, queue)
        service = JobService(
            SqlAlchemyJobRepository(session),
            SqlAlchemyVideoRepository(session),
            queue,
            tracker,
        )
        service.submit(video_with_candidates.id, 0)

    # Nothing had been published at the moment of commit.
    assert tracker.enqueued_at_commit == []
    assert len(queue.enqueued) == 1


def test_submitting_an_unknown_candidate_is_rejected(settings, video_with_candidates):
    queue = RecordingQueue()
    with session_scope() as session:
        service = JobService(
            SqlAlchemyJobRepository(session),
            SqlAlchemyVideoRepository(session),
            queue,
            _CommitTracker(session, queue),
        )
        with pytest.raises(UnknownCandidate):
            service.submit(video_with_candidates.id, 7)

    assert queue.enqueued == []


def test_submitting_before_candidates_exist_is_rejected(settings, ingest_video):
    video = ingest_video(seconds=1.0)
    queue = RecordingQueue()
    with session_scope() as session:
        videos = SqlAlchemyVideoRepository(session)
        service = JobService(
            SqlAlchemyJobRepository(session), videos, queue, _CommitTracker(session, queue)
        )
        with pytest.raises(UnknownCandidate):
            service.submit(video.id, 0)


# ------------------------------------------------------------- restart recovery


def _job(job_id: str, status: JobStatus, video_id: str) -> JobRecord:
    return JobRecord(
        id=job_id,
        video_id=video_id,
        kind=JobKind.ANALYSIS,
        candidate_index=0,
        status=status,
        progress=0.5 if status is JobStatus.RUNNING else 0.0,
        created_at=datetime.now(timezone.utc),
    )


def test_restart_fails_running_jobs_and_requeues_queued_ones(settings, video_with_candidates):
    queue = RecordingQueue()
    video_id = video_with_candidates.id
    with session_scope() as session:
        jobs = SqlAlchemyJobRepository(session)
        jobs.add(_job("was-running", JobStatus.RUNNING, video_id))
        jobs.add(_job("was-queued", JobStatus.QUEUED, video_id))

        recover_unfinished_jobs(jobs, queue)

        # The process that owned the running job is gone; leaving the row at
        # "running" would mean a status that never changes again.
        assert jobs.get("was-running").status is JobStatus.FAILED
        assert "restart" in jobs.get("was-running").error
        assert jobs.get("was-queued").status is JobStatus.QUEUED

    assert queue.enqueued == ["was-queued"]
