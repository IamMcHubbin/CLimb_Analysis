"""The analysis job end to end, with a stub pose model."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.analysis import AnalysisJobHandler
from app.candidates import CandidateService
from app.db import SqlAlchemyJobRepository, SqlAlchemyVideoRepository, session_scope
from app.db.repository import JobStatus
from app.jobs.service import JobService
from app.keypoints import ParquetKeypointStore
from app.pose.base import Landmark, PersonPose

from tests.test_candidates import FakeEstimator, _person
from tests.test_jobs import RecordingQueue, _CommitTracker

LANDMARK_NAMES = ("a", "b", "c", "d")
CONNECTIONS = ((0, 1), (1, 2), (2, 3))


class ScriptedEstimator(FakeEstimator):
    """Returns whatever the script says for each successive frame."""

    def __init__(self, per_frame):
        super().__init__([])
        self._per_frame = per_frame

    @property
    def landmark_names(self):
        return LANDMARK_NAMES

    @property
    def landmark_connections(self):
        return CONNECTIONS

    def detect(self, frame, timestamp_ms: int = 0):
        index = self.calls
        self.calls += 1
        if index < len(self._per_frame):
            return tuple(self._per_frame[index])
        return ()


def _four_point_person(x: float, y: float, size: float = 0.2) -> PersonPose:
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    return PersonPose(
        landmarks=tuple(
            Landmark(x=cx, y=cy, z=0.0, visibility=0.9, presence=0.9) for cx, cy in corners
        )
    )


@pytest.fixture
def prepared(settings, ingest_video):
    """A video with a stored candidate set and a queued job, ready to run."""
    video = ingest_video(seconds=1.0)  # 30 frames; candidate frame is 15
    queue = RecordingQueue()
    with session_scope() as session:
        videos = SqlAlchemyVideoRepository(session)

        @contextmanager
        def factory():
            yield FakeEstimator([_person(0.4, 0.4, 0.2)])

        CandidateService(videos, settings, estimator_factory=factory).detect(video)
        service = JobService(
            SqlAlchemyJobRepository(session), videos, queue, _CommitTracker(session, queue)
        )
        job = service.submit(video.id, 0)
    return video, job


def _handler(settings, per_frame):
    estimator = ScriptedEstimator(per_frame)

    @contextmanager
    def factory():
        yield estimator

    return AnalysisJobHandler(settings, estimator_factory=factory), estimator


def test_a_successful_run_stores_keypoints_and_marks_the_job_done(settings, prepared):
    video, job = prepared
    # The same person, in the same place, in every frame.
    handler, estimator = _handler(settings, [[_four_point_person(0.4, 0.4)]] * video.frame_count)

    handler(job.id)

    with session_scope() as session:
        finished = SqlAlchemyJobRepository(session).get(job.id)
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    assert finished.status is JobStatus.DONE
    assert finished.progress == 1.0
    assert finished.error is None
    assert stored.keypoints_path is not None

    data = ParquetKeypointStore(settings).read(stored.keypoints_path)
    assert data.metadata.landmark_names == LANDMARK_NAMES
    assert data.metadata.landmark_connections == CONNECTIONS
    assert data.tracked_frame_count == video.frame_count
    assert data.gap_frame_count == 0
    # Two passes over the clip: one to find and track, one to refine the
    # landmarks on a crop. One inference per frame in each.
    assert estimator.calls == video.frame_count * 2


def test_refinement_can_be_turned_off(settings, prepared):
    """Without it, one pass and one inference per frame."""
    import dataclasses

    video, job = prepared
    settings = dataclasses.replace(settings, refine_landmarks=False)
    handler, estimator = _handler(settings, [[_four_point_person(0.4, 0.4)]] * video.frame_count)

    handler(job.id)

    assert estimator.calls == video.frame_count
    with session_scope() as session:
        assert SqlAlchemyJobRepository(session).get(job.id).status is JobStatus.DONE


def test_a_failed_refinement_keeps_the_first_pass_track(settings, prepared):
    """An improvement that fails should not lose the answer it was improving."""
    from contextlib import contextmanager

    video, job = prepared
    calls = {"n": 0}

    @contextmanager
    def factory():
        calls["n"] += 1
        if calls["n"] > 1:                       # the refinement pass
            raise RuntimeError("crop model fell over")
        yield ScriptedEstimator([[_four_point_person(0.4, 0.4)]] * video.frame_count)

    AnalysisJobHandler(settings, estimator_factory=factory)(job.id)

    with session_scope() as session:
        assert SqlAlchemyJobRepository(session).get(job.id).status is JobStatus.DONE
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    assert stored.keypoints_path is not None
    data = ParquetKeypointStore(settings).read(stored.keypoints_path)
    assert data.tracked_frame_count == video.frame_count


def test_frames_before_the_seed_frame_are_tracked_too(settings, prepared):
    """The user picks a person mid-clip; the start of the clip still gets a track."""
    video, job = prepared
    handler, _ = _handler(settings, [[_four_point_person(0.4, 0.4)]] * video.frame_count)

    handler(job.id)

    with session_scope() as session:
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    data = ParquetKeypointStore(settings).read(stored.keypoints_path)

    assert 0 in data.frames
    assert 1 in data.frames


def test_unmatched_frames_become_gaps(settings, prepared):
    video, job = prepared
    # The tracked person vanishes for frames 20-24, replaced by someone
    # elsewhere in the frame who must not be picked up instead.
    script = []
    for index in range(video.frame_count):
        if 20 <= index < 25:
            script.append([_four_point_person(0.85, 0.85, 0.1)])
        else:
            script.append([_four_point_person(0.4, 0.4)])
    handler, _ = _handler(settings, script)

    handler(job.id)

    with session_scope() as session:
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    data = ParquetKeypointStore(settings).read(stored.keypoints_path)

    assert data.gap_frame_count == 5
    for index in range(20, 25):
        assert index not in data.frames
    # And the track resumes on the far side of the hole.
    assert 25 in data.frames


def test_a_failure_is_recorded_on_the_job(settings, prepared):
    video, job = prepared

    @contextmanager
    def exploding():
        raise RuntimeError("the model fell over")
        yield  # pragma: no cover

    handler = AnalysisJobHandler(settings, estimator_factory=exploding)
    handler(job.id)

    with session_scope() as session:
        failed = SqlAlchemyJobRepository(session).get(job.id)
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    assert failed.status is JobStatus.FAILED
    assert "fell over" in failed.error
    assert failed.finished_at is not None
    # No half-written track is left pointing at nothing.
    assert stored.keypoints_path is None


def test_progress_advances_while_running(settings, prepared):
    video, job = prepared
    seen: list[float] = []

    handler, _ = _handler(settings, [[_four_point_person(0.4, 0.4)]] * video.frame_count)
    original = handler._report_progress

    def spy(job_id: str, fraction: float) -> None:
        seen.append(fraction)
        original(job_id, fraction)

    handler._report_progress = spy
    handler(job.id)

    assert seen, "progress was never reported"
    assert seen == sorted(seen)
    assert max(seen) < 1.0  # 1.0 is only set when the job is marked done
