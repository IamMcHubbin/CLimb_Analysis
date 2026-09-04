"""The properties the run model exists to guarantee.

These cover the seam between the two designs: immutable runs on one side,
asynchronous ingest, refinement and retention on the other.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.analysis import AnalysisJobHandler
from app.candidates import CandidateService
from app.db import (
    SqlAlchemyAnalysisRunRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyVideoRepository,
    session_scope,
)
from app.keypoints import ParquetKeypointStore
from app.retention import FootageRetention

from tests.test_analysis import ScriptedEstimator, _four_point_person
from tests.test_analysis_runs import _submit
from tests.test_candidates import FakeEstimator, _person


def _scripted(person, frames):
    @contextmanager
    def factory():
        yield ScriptedEstimator([[person]] * frames)
    return factory


def _repick(settings, video, box):
    """Re-run the candidate picker, replacing the stored candidate set."""
    @contextmanager
    def factory():
        yield FakeEstimator([_person(*box)])

    with session_scope() as session:
        CandidateService(
            SqlAlchemyVideoRepository(session), settings, estimator_factory=factory
        ).detect(video, frame_index=3)


# ------------------------------------------- immutable candidate selection


def test_repicking_after_submission_does_not_change_a_queued_run(settings, ingest_video):
    """The bug the run model closes.

    Candidates live on the video and are replaced wholesale when the picker
    runs again. A worker that read them at execution time would analyse
    whoever now sits at the chosen index.
    """
    video, job, _ = _submit(settings, ingest_video)
    with session_scope() as session:
        original = SqlAlchemyAnalysisRunRepository(session).get(job.analysis_run_id).seed_box

    _repick(settings, video, (0.75, 0.75, 0.1))

    with session_scope() as session:
        after = SqlAlchemyAnalysisRunRepository(session).get(job.analysis_run_id).seed_box
    assert after == original, "the run's seed must not follow the candidate set"


def test_the_worker_tracks_the_seed_on_the_run(settings, ingest_video):
    video, job, _ = _submit(settings, ingest_video)
    _repick(settings, video, (0.75, 0.75, 0.1))

    # The estimator reports a person where the *original* candidate was. If
    # the worker used the replaced candidate set it would find no match and
    # produce nothing but gaps.
    AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )(job.id)

    with session_scope() as session:
        run = SqlAlchemyAnalysisRunRepository(session).get(job.analysis_run_id)
    data = ParquetKeypointStore(settings).read(run.keypoints_path)
    assert data.tracked_frame_count == video.frame_count
    assert data.gap_frame_count == 0


def test_a_job_with_no_run_fails_rather_than_guessing(settings, ingest_video):
    """The shape a job left by the pre-run schema has after migration.

    Deleting a run cascades to its jobs, so the only way to reach a job
    without one is to have been created before runs existed. It must fail
    loudly rather than fall back to the live candidate set.
    """
    from sqlalchemy import text

    video, job, _ = _submit(settings, ingest_video)
    with session_scope() as session:
        session.execute(
            text("UPDATE jobs SET analysis_run_id = NULL WHERE id = :i"), {"i": job.id}
        )

    AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )(job.id)

    with session_scope() as session:
        from app.db.repository import JobStatus
        failed = SqlAlchemyJobRepository(session).get(job.id)
    assert failed.status is JobStatus.FAILED
    assert "analysis run" in failed.error


def test_deleting_a_run_takes_its_jobs_with_it(settings, ingest_video):
    """Recorded because it is load-bearing: a job without its run is inert."""
    from sqlalchemy import text

    _, job, _ = _submit(settings, ingest_video)
    with session_scope() as session:
        session.execute(
            text("DELETE FROM analysis_runs WHERE id = :i"), {"i": job.analysis_run_id}
        )

    with session_scope() as session:
        assert SqlAlchemyJobRepository(session).get(job.id) is None


# ----------------------------------------------- independent run artifacts


def test_two_runs_write_separate_artifacts_and_neither_is_overwritten(settings, ingest_video):
    video, first, queue = _submit(settings, ingest_video)
    _repick(settings, video, (0.6, 0.6, 0.2))
    _, second, _ = _submit(settings, ingest_video, video=video, queue=queue)

    handler = AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )
    handler(first.id)
    with session_scope() as session:
        first_path = SqlAlchemyAnalysisRunRepository(session).get(first.analysis_run_id).keypoints_path
    handler(second.id)

    with session_scope() as session:
        runs = SqlAlchemyAnalysisRunRepository(session)
        second_path = runs.get(second.analysis_run_id).keypoints_path
        # The first run's artifact is untouched by the second analysis.
        assert runs.get(first.analysis_run_id).keypoints_path == first_path
        latest = SqlAlchemyVideoRepository(session).get(video.id).keypoints_path

    assert first_path != second_path
    assert settings.resolve(first_path).exists()
    assert settings.resolve(second_path).exists()
    # The video points at whichever finished last - a convenience, not the
    # canonical location.
    assert latest == second_path


# --------------------------------------------- refinement uses the run model


def test_the_refinement_pass_uses_the_run_model(settings, ingest_video):
    """Both passes must come from the run, not from whatever is set now."""
    import dataclasses

    submitted_with = dataclasses.replace(settings, pose_model="heavy")
    video, job, _ = _submit(submitted_with, ingest_video)

    seen: list[str] = []

    def factory(variant=None, max_people=None):
        seen.append(variant)

        @contextmanager
        def opened():
            yield ScriptedEstimator([[_four_point_person(0.2, 0.2)]] * video.frame_count)

        return opened()

    handler = AnalysisJobHandler(dataclasses.replace(settings, pose_model="lite"))
    handler._default_estimator = factory        # stand in for the real model
    handler(job.id)

    assert seen == ["heavy", "heavy"], "tracking and refinement both take the run's model"


# --------------------------------------------------- retention of artifacts


def test_retention_never_sweeps_a_referenced_artifact(settings, ingest_video):
    video, job, _ = _submit(settings, ingest_video)
    AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )(job.id)

    with session_scope() as session:
        runs = SqlAlchemyAnalysisRunRepository(session)
        path = runs.get(job.analysis_run_id).keypoints_path
        # Well past the grace period, so only the reference protects it.
        removed = FootageRetention(settings).sweep_orphan_artifacts(
            SqlAlchemyVideoRepository(session), runs,
            now=datetime.now(timezone.utc) + timedelta(days=30),
        )

    assert removed == 0
    assert settings.resolve(path).exists()


def test_retention_sweeps_an_artifact_no_run_refers_to(settings, ingest_video):
    video, job, _ = _submit(settings, ingest_video)
    AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )(job.id)

    # A file left behind by a job that died before recording its path.
    orphan = settings.keypoints_dir / "abandoned.parquet"
    orphan.write_bytes(b"not really parquet, but a file nothing points at")

    with session_scope() as session:
        removed = FootageRetention(settings).sweep_orphan_artifacts(
            SqlAlchemyVideoRepository(session),
            SqlAlchemyAnalysisRunRepository(session),
            now=datetime.now(timezone.utc) + timedelta(days=30),
        )
        kept = SqlAlchemyAnalysisRunRepository(session).get(job.analysis_run_id).keypoints_path

    assert removed == 1
    assert not orphan.exists()
    assert settings.resolve(kept).exists()


def test_a_freshly_written_artifact_is_not_mistaken_for_an_orphan(settings):
    """A worker writes the file before it records the path on the run."""
    settings.keypoints_dir.mkdir(parents=True, exist_ok=True)
    just_written = settings.keypoints_dir / "inflight.parquet"
    just_written.write_bytes(b"mid-write")

    with session_scope() as session:
        removed = FootageRetention(settings).sweep_orphan_artifacts(
            SqlAlchemyVideoRepository(session), SqlAlchemyAnalysisRunRepository(session)
        )

    assert removed == 0
    assert just_written.exists()


def test_deleting_footage_leaves_run_artifacts_alone(settings, ingest_video):
    """Keypoints outlive the clip; that promise still holds per run."""
    video, job, _ = _submit(settings, ingest_video)
    AnalysisJobHandler(
        settings, estimator_factory=_scripted(_four_point_person(0.2, 0.2), video.frame_count)
    )(job.id)

    with session_scope() as session:
        videos = SqlAlchemyVideoRepository(session)
        runs = SqlAlchemyAnalysisRunRepository(session)
        path = runs.get(job.analysis_run_id).keypoints_path
        FootageRetention(settings).delete_footage(videos, videos.get(video.id))

    assert settings.resolve(path).exists()
    with session_scope() as session:
        assert not SqlAlchemyVideoRepository(session).get(video.id).has_footage
