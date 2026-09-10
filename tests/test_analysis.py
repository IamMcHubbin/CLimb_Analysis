"""The analysis job end to end, with a stub pose model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from app.analysis import AnalysisJobHandler
from app.candidates import CandidateService
from app.db import SqlAlchemyAnalysisRunRepository, SqlAlchemyJobRepository, SqlAlchemyVideoRepository, session_scope
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
            SqlAlchemyJobRepository(session), videos, queue, _CommitTracker(session, queue),
            SqlAlchemyAnalysisRunRepository(session), settings=settings,
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


def test_refinement_follows_the_run_not_current_settings(settings, ingest_video):
    """Turning refinement off after submission must not change a queued run.

    The flag is snapshotted onto the AnalysisRun, so the switch has to be off
    when the analysis is *asked for*, not when it happens.
    """
    import dataclasses

    from tests.test_analysis_runs import _submit

    disabled = dataclasses.replace(settings, refine_landmarks=False)
    video, job, _ = _submit(disabled, ingest_video)
    # Global settings say refine; the run, submitted with it off, must win.
    handler, estimator = _handler(settings, [[_four_point_person(0.2, 0.2)]] * video.frame_count)

    handler(job.id)

    assert estimator.calls == video.frame_count, "one pass only"
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


# ------------------------------------------------ reusing the tracked box

class RoiAwareEstimator(ScriptedEstimator):
    """Records the region it was asked to pose, and honours the script.

    Stands in for a top-down model: given a region it poses that region, and
    given none it searches. What it returns does not depend on the region -
    the point of these tests is which regions it is handed, and what the
    caller does with a reply it should not trust.
    """

    def __init__(self, per_frame, confidence: float = 0.9):
        super().__init__(per_frame)
        self.rois: list[object] = []
        self._confidence = confidence

    @property
    def uses_roi_hint(self) -> bool:
        return True

    def detect(self, frame, timestamp_ms: int = 0, *, roi=None):
        self.rois.append(roi)
        people = super().detect(frame, timestamp_ms)
        if self._confidence >= 0.9:
            return people
        return tuple(
            PersonPose(
                landmarks=tuple(
                    Landmark(x=lm.x, y=lm.y, z=lm.z,
                             visibility=self._confidence, presence=self._confidence)
                    for lm in person.landmarks
                )
            )
            for person in people
        )


def _roi_handler(settings, per_frame, confidence: float = 0.9):
    estimator = RoiAwareEstimator(per_frame, confidence)

    @contextmanager
    def factory():
        yield estimator

    return AnalysisJobHandler(settings, estimator_factory=factory), estimator


def test_the_tracked_box_is_not_reused_unless_it_is_turned_on(settings, prepared):
    video, job = prepared
    handler, estimator = _roi_handler(
        settings, [[_four_point_person(0.4, 0.4)]] * video.frame_count
    )

    handler(job.id)

    assert estimator.rois, "the estimator was never called"
    assert all(roi is None for roi in estimator.rois)


def test_reusing_the_box_still_searches_whole_frames_periodically(settings, prepared):
    """Every Nth frame must be an independent look, or a drifted track is invisible."""
    video, job = prepared
    tuned = replace(settings, reuse_tracked_box=True, reanchor_frames=4)
    handler, estimator = _roi_handler(
        tuned, [[_four_point_person(0.4, 0.4)]] * video.frame_count
    )

    handler(job.id)

    searched = sum(1 for roi in estimator.rois if roi is None)
    posed = sum(1 for roi in estimator.rois if roi is not None)
    assert posed > 0, "the hint was never used, so nothing was saved"
    assert searched > 0, "every frame used the hint, so a lost track could never be noticed"


def test_frames_before_the_seed_never_use_a_hint(settings, prepared):
    """They are tracked backwards afterwards, so nothing yet knows where to look."""
    video, job = prepared
    tuned = replace(settings, reuse_tracked_box=True, reanchor_frames=1000)
    handler, estimator = _roi_handler(
        tuned, [[_four_point_person(0.4, 0.4)]] * video.frame_count
    )

    handler(job.id)

    seed = video.frame_count // 2
    assert all(roi is None for roi in estimator.rois[: seed + 1])


def test_an_unconvincing_posed_region_becomes_a_gap(settings, prepared):
    """A pose taken from the tracked box always fits it, so position cannot
    reject it. Confidence is the only thing left that says the climber has
    gone and the model is describing an empty wall."""
    video, job = prepared
    tuned = replace(
        settings, reuse_tracked_box=True, reanchor_frames=1000, roi_min_visibility=0.8
    )
    handler, _ = _roi_handler(
        tuned, [[_four_point_person(0.4, 0.4)]] * video.frame_count, confidence=0.1
    )

    handler(job.id)

    with session_scope() as session:
        stored = SqlAlchemyVideoRepository(session).get(video.id)
    data = ParquetKeypointStore(tuned).read(stored.keypoints_path)
    seed = video.frame_count // 2
    # A gap is an absent row, so every frame past the seed should be missing.
    after_seed = [index for index in range(seed + 1, video.frame_count)]
    assert after_seed, "expected frames after the seed"
    assert all(index not in data.frames for index in after_seed)
