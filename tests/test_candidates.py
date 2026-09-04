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
def video(settings, ingest_video):
    """A real normalised video row, so frame reading is genuine."""
    return ingest_video(seconds=2.0)


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


class SometimesEmptyEstimator(FakeEstimator):
    """Finds nobody until it has been asked ``empty_for`` times.

    Stands in for the clip where the middle frame is exactly where somebody
    walked through the shot.
    """

    def __init__(self, people, empty_for: int):
        super().__init__(people)
        self._empty_for = empty_for

    def detect(self, frame, timestamp_ms: int = 0):
        self.calls += 1
        if self.calls <= self._empty_for:
            return ()
        return tuple(self.people)


def test_other_frames_are_tried_when_the_middle_has_nobody(settings, repository, video):
    estimator = SometimesEmptyEstimator([_person(0.1, 0.1, 0.2)], empty_for=2)
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video)

    assert estimator.calls == 3
    assert len(result.candidates) == 1
    # It reports the frame it actually found somebody in, not the one it wanted.
    assert result.frame_index != service.default_frame_index(video)
    assert service.frame_path(video).exists()


def test_search_gives_up_and_reports_the_first_frame(settings, repository, video):
    """A clip with nobody in it still yields a frame the client can show."""
    estimator = FakeEstimator([])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video)

    assert result.candidates == ()
    assert result.frame_index == service.default_frame_index(video)
    assert estimator.calls > 1, "should have tried more than one frame"
    assert service.frame_path(video).exists()


def test_an_explicit_frame_is_never_second_guessed(settings, repository, video):
    """The caller is steering; answering about a different frame would be worse."""
    estimator = FakeEstimator([])
    service = CandidateService(repository, settings, estimator_factory=_factory(estimator))

    result = service.detect(video, frame_index=7)

    assert result.frame_index == 7
    assert result.candidates == ()
    assert estimator.calls == 1


def test_search_order_is_spread_across_the_clip(settings, repository, video):
    service = CandidateService(repository, settings, estimator_factory=_factory(FakeEstimator([])))

    order = service._search_order(video)

    assert order[0] == service.default_frame_index(video)
    assert len(order) == len(set(order)), "no frame tried twice"
    assert all(0 <= index < video.frame_count for index in order)
    # Spread widely enough that a two-second occlusion cannot swallow them all.
    assert max(order) - min(order) > video.frame_count // 2
