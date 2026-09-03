"""Candidate detection: the numbering the user's choice depends on."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from app.candidates import CandidateService
from app.db import SqlAlchemyVideoRepository, session_scope
from app.db.repository import VideoRecord
from app.geometry import BoundingBox
from app.pose.base import Landmark, PersonPose


def _person(x: float, y: float, size: float) -> PersonPose:
    """A person whose landmarks span a square box of the given size."""
    corners = [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]
    return PersonPose(
        landmarks=tuple(
            Landmark(x=cx, y=cy, z=0.0, visibility=0.9, presence=0.9) for cx, cy in corners
        )
    )


class FakeEstimator:
    """Returns a fixed set of people, in a deliberately unhelpful order."""

    def __init__(self, people):
        self.people = people
        self.calls = 0

    def detect(self, frame, timestamp_ms: int = 0):
        self.calls += 1
        return tuple(self.people)

    def close(self):
        pass


def _factory(estimator):
    @contextmanager
    def factory():
        yield estimator

    return factory


@pytest.fixture
def video(settings, make_video):
    """A real normalised video row, so frame reading is genuine."""
    from app.ingest.service import IngestService

    source = make_video(seconds=2.0)
    with session_scope() as session:
        service = IngestService(SqlAlchemyVideoRepository(session), settings=settings)
        return service.ingest(source, "clip.mp4")


@pytest.fixture
def repository():
    with session_scope() as session:
        yield SqlAlchemyVideoRepository(session)


def test_detects_in_the_middle_of_the_clip(settings, repository, video):
    estimator = FakeEstimator([_person(0.1, 0.1, 0.2)])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video)

    assert result.frame_index == video.frame_count // 2
    assert len(result.candidates) == 1
    # The frame is cached as JPEG for the client to draw boxes on.
    assert service.frame_path(video).exists()
    assert service.frame_path(video).read_bytes().startswith(b"\xff\xd8")


def test_candidates_are_numbered_largest_first(settings, repository, video):
    # Deliberately out of order: small, large, medium.
    estimator = FakeEstimator([
        _person(0.8, 0.8, 0.1),
        _person(0.0, 0.0, 0.6),
        _person(0.4, 0.4, 0.3),
    ])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video)

    assert [candidate.index for candidate in result.candidates] == [0, 1, 2]
    areas = [candidate.bounding_box.area for candidate in result.candidates]
    assert areas == sorted(areas, reverse=True)


def test_candidate_set_is_reused_rather_than_renumbered(settings, repository, video):
    estimator = FakeEstimator([_person(0.1, 0.1, 0.2)])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    first = service.get_or_detect(video)
    second = service.get_or_detect(video)

    # Detection ran once; the second call came from storage. Re-running would
    # renumber the candidates under a user who is mid-choice.
    assert estimator.calls == 1
    assert first == second


def test_asking_for_another_frame_re_detects(settings, repository, video):
    estimator = FakeEstimator([_person(0.1, 0.1, 0.2)])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    service.get_or_detect(video)
    other = service.get_or_detect(video, frame_index=3)

    assert estimator.calls == 2
    assert other.frame_index == 3


def test_frame_index_is_clamped_to_the_clip(settings, repository, video):
    estimator = FakeEstimator([])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video, frame_index=10_000)

    assert result.frame_index == video.frame_count - 1


def test_candidate_set_survives_a_round_trip(settings, repository, video):
    estimator = FakeEstimator([_person(0.2, 0.3, 0.4)])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))
    stored = service.detect(video)

    reloaded = repository.get_candidates(video.id)

    assert reloaded == stored
    assert reloaded.get(0).bounding_box == stored.candidates[0].bounding_box
    assert reloaded.get(99) is None


def test_boxes_are_normalised_and_clamped(settings, repository, video):
    # Landmarks outside the frame: MediaPipe extrapolates occluded joints.
    estimator = FakeEstimator([_person(-0.5, -0.5, 3.0)])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    box = service.detect(video).candidates[0].bounding_box

    assert box == BoundingBox(0.0, 0.0, 1.0, 1.0)


def test_video_with_no_frames_is_rejected(settings, repository, video):
    empty = VideoRecord(
        **{**video.__dict__, "id": "empty", "frame_count": 0, "created_at": datetime.now(timezone.utc)}
    )
    service = CandidateService(repository, settings, estimator_factory=_factory(FakeEstimator([])))
    with pytest.raises(ValueError):
        service.detect(empty)
