from __future__ import annotations

from contextlib import contextmanager

from app.analysis import AnalysisJobHandler
from app.candidates import CandidateService
from app.db import (
    SqlAlchemyAnalysisRunRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyVideoRepository,
    session_scope,
)
from app.ingest.service import IngestService
from app.jobs.service import JobService
from app.keypoints import ParquetKeypointStore
from tests.test_analysis import ScriptedEstimator, _four_point_person
from tests.test_candidates import FakeEstimator, _person
from tests.test_jobs import RecordingQueue, _CommitTracker


def _submit(settings, make_video):
    source = make_video(seconds=1.0)
    with session_scope() as session:
        videos = SqlAlchemyVideoRepository(session)
        video = IngestService(videos, settings=settings).ingest(source, "clip.mp4")

        @contextmanager
        def factory():
            yield FakeEstimator([_person(0.2, 0.2, 0.2)])

        CandidateService(videos, settings, estimator_factory=factory).detect(video)
        queue = RecordingQueue()
        job_service = JobService(
            SqlAlchemyJobRepository(session), videos, queue, _CommitTracker(session, queue),
            SqlAlchemyAnalysisRunRepository(session), settings=settings,
        )
        return video, job_service.submit(video.id, 0), queue


def test_two_runs_have_independent_snapshots_and_artifacts(settings, make_video):
    video, first_job, queue = _submit(settings, make_video)
    with session_scope() as session:
        videos = SqlAlchemyVideoRepository(session)
        runs = SqlAlchemyAnalysisRunRepository(session)
        # Replacing candidates must not mutate the first run's seed.
        CandidateService(
            videos, settings,
            estimator_factory=lambda: _factory(FakeEstimator([_person(0.7, 0.7, 0.1)])),
        ).detect(video, frame_index=3)
        second_job = JobService(
            SqlAlchemyJobRepository(session), videos, queue, _CommitTracker(session, queue),
            runs, settings=settings,
        ).submit(video.id, 0)
        first_run = runs.get(first_job.analysis_run_id)
        second_run = runs.get(second_job.analysis_run_id)

    assert first_run.id != second_run.id
    assert first_run.seed_box != second_run.seed_box

    old_person = [_four_point_person(0.2, 0.2)] * video.frame_count
    first_handler = AnalysisJobHandler(settings, estimator_factory=lambda: _scripted(old_person))
    first_handler(first_job.id)
    second_handler = AnalysisJobHandler(settings, estimator_factory=lambda: _scripted(old_person))
    second_handler(second_job.id)

    with session_scope() as session:
        runs = SqlAlchemyAnalysisRunRepository(session)
        first_run = runs.get(first_job.analysis_run_id)
        second_run = runs.get(second_job.analysis_run_id)
    assert first_run.keypoints_path != second_run.keypoints_path
    assert settings.resolve(first_run.keypoints_path).exists()
    assert settings.resolve(second_run.keypoints_path).exists()
    assert ParquetKeypointStore(settings).read(first_run.keypoints_path).tracked_frame_count == video.frame_count


def _factory(estimator):
    @contextmanager
    def factory():
        yield estimator
    return factory


def _scripted(people):
    @contextmanager
    def factory():
        yield ScriptedEstimator([[person] for person in people])
    return factory
