"""Footage retention: the promise that clips do not linger."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from app.candidates import CandidateService
from app.db import SqlAlchemyVideoRepository, session_scope
from app.retention import FootageRetention, RetentionJanitor

from tests.test_candidates import FakeEstimator, _person


@pytest.fixture
def repository():
    with session_scope() as session:
        yield SqlAlchemyVideoRepository(session)


@pytest.fixture
def video(settings, ingest_video, repository):
    ready = ingest_video(seconds=1.0)
    # Re-read through this test's session so later writes here are visible.
    return repository.get(ready.id)


def _file(settings, record):
    return settings.resolve(record.stored_path)


def test_deleting_footage_removes_the_file_but_keeps_the_row(settings, repository, video):
    assert _file(settings, video).exists()

    deleted = FootageRetention(settings).delete_footage(repository, video)

    assert deleted is True
    assert not _file(settings, video).exists()
    after = repository.get(video.id)
    assert after is not None, "the row must outlive the footage"
    assert after.has_footage is False
    assert after.footage_deleted_at is not None


def test_the_candidate_frame_goes_with_the_footage(settings, repository, video):
    """It is a still of the same clip, so it cannot be left behind."""

    @contextmanager
    def factory():
        yield FakeEstimator([_person(0.1, 0.1, 0.3)])

    service = CandidateService(repository, settings, estimator_factory=factory)
    service.detect(video)
    assert service.frame_path(video).exists()

    FootageRetention(settings).delete_footage(repository, video)

    assert not service.frame_path(video).exists()


def test_deleting_twice_is_not_an_error(settings, repository, video):
    retention = FootageRetention(settings)
    assert retention.delete_footage(repository, video) is True
    assert retention.delete_footage(repository, repository.get(video.id)) is False


def test_analysed_footage_survives_until_its_window_passes(settings, repository, video):
    settings = dataclasses.replace(settings, retain_analysed_seconds=3600)
    repository.set_latest_keypoints_path(video.id, "keypoints/x.parquet")
    retention = FootageRetention(settings)

    assert retention.sweep(repository) == 0
    assert repository.get(video.id).has_footage

    # An hour and a minute later it is due.
    later = datetime.now(timezone.utc) + timedelta(seconds=3660)
    assert retention.sweep(repository, now=later) == 1
    assert not repository.get(video.id).has_footage


def test_zero_retention_deletes_as_soon_as_it_is_analysed(settings, repository, video):
    settings = dataclasses.replace(settings, retain_analysed_seconds=0)
    repository.set_latest_keypoints_path(video.id, "keypoints/x.parquet")

    assert FootageRetention(settings).sweep(repository) == 1
    assert not repository.get(video.id).has_footage


def test_unanalysed_uploads_are_swept_on_their_own_schedule(settings, repository, video):
    """An upload nobody analysed is abandoned, not in use."""
    settings = dataclasses.replace(
        settings, retain_analysed_seconds=0, retain_unanalysed_seconds=86400
    )
    retention = FootageRetention(settings)

    # Not analysed, so the short analysed-window does not apply to it.
    assert retention.sweep(repository) == 0
    assert repository.get(video.id).has_footage

    tomorrow = datetime.now(timezone.utc) + timedelta(seconds=86500)
    assert retention.sweep(repository, now=tomorrow) == 1


def test_sweep_skips_footage_already_deleted(settings, repository, video):
    settings = dataclasses.replace(settings, retain_analysed_seconds=0)
    repository.set_latest_keypoints_path(video.id, "keypoints/x.parquet")
    retention = FootageRetention(settings)

    assert retention.sweep(repository) == 1
    assert retention.sweep(repository) == 0


def test_expiry_is_reported_from_the_right_moment(settings, repository, video):
    settings = dataclasses.replace(
        settings, retain_analysed_seconds=3600, retain_unanalysed_seconds=86400
    )
    retention = FootageRetention(settings)

    # Before analysis, counted from upload.
    assert retention.expires_at(video) == video.created_at + timedelta(seconds=86400)

    repository.set_latest_keypoints_path(video.id, "keypoints/x.parquet")
    analysed = repository.get(video.id)
    assert retention.expires_at(analysed) == analysed.analysis_completed_at + timedelta(seconds=3600)

    FootageRetention(settings).delete_footage(repository, analysed)
    assert retention.expires_at(repository.get(video.id)) is None


def test_the_janitor_sweeps_and_stops(settings, ingest_video):
    """The janitor opens its own session, so it only sees committed rows."""
    settings = dataclasses.replace(settings, retain_analysed_seconds=0)

    video = ingest_video(seconds=1.0)
    with session_scope() as session:
        SqlAlchemyVideoRepository(session).set_latest_keypoints_path(video.id, "keypoints/x.parquet")
    # Committed on leaving the block above; before that the janitor's session
    # could not have seen this video at all.

    janitor = RetentionJanitor(FootageRetention(settings), interval_seconds=1)
    assert janitor.sweep_once() == 1
    janitor.start()
    janitor.stop(timeout=5)

    with session_scope() as session:
        assert not SqlAlchemyVideoRepository(session).get(video.id).has_footage
