"""Crop-based landmark refinement."""

from __future__ import annotations

import pytest

from app.frames import FrameReader
from app.geometry import BoundingBox
from app.pose.base import Landmark, PersonPose
from app.pose.refine import refine_landmarks
from app.tracking import TrackedFrame

from tests.test_candidates import FakeEstimator


def _person(x: float, y: float, size: float = 0.2) -> PersonPose:
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    return PersonPose(
        landmarks=tuple(
            Landmark(x=cx, y=cy, z=0.0, visibility=0.9, presence=0.9) for cx, cy in corners
        )
    )


class CropRecordingEstimator(FakeEstimator):
    """Returns a fixed pose and remembers the size of every image it saw."""

    def __init__(self, people, sizes=None):
        super().__init__(people)
        self.sizes = sizes if sizes is not None else []

    def detect(self, frame, timestamp_ms: int = 0):
        self.calls += 1
        self.sizes.append(frame.shape[:2])
        return tuple(self.people)


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize(
    ("box", "margin"),
    [(BoundingBox(0.4, 0.4, 0.1, 0.2), 0.5), (BoundingBox(0.35, 0.3, 0.3, 0.1), 0.55)],
)
def test_expanded_crop_is_square_and_contains_the_box(box, margin):
    """Square, because the model letterboxes anything else and wastes the pixels."""
    grown = box.square_expanded(margin)
    assert grown.x <= box.x and grown.y <= box.y
    assert grown.x2 >= box.x2 and grown.y2 >= box.y2
    assert grown.width == pytest.approx(grown.height, abs=1e-9)


def test_clamping_at_an_edge_wins_over_staying_square():
    """Sampling outside the image is worse than an off-square crop."""
    grown = BoundingBox(0.0, 0.0, 0.3, 0.1).square_expanded(0.55)
    assert grown.x == 0.0 and grown.y == 0.0
    assert grown.width != pytest.approx(grown.height)


def test_expanded_crop_is_clamped_to_the_frame():
    grown = BoundingBox(0.9, 0.9, 0.1, 0.1).square_expanded(1.0)
    assert grown.x >= 0.0 and grown.y >= 0.0
    assert grown.x2 <= 1.0 and grown.y2 <= 1.0


# ---------------------------------------------------------------- refinement


@pytest.fixture
def reader(settings, ingest_video):
    video = ingest_video(seconds=1.0, width=640, height=480)
    with FrameReader(settings.resolve(video.stored_path)) as opened:
        yield opened


def test_the_model_is_shown_a_crop_not_the_whole_frame(reader):
    """The point of the exercise: more pixels of climber per pixel of input."""
    estimator = CropRecordingEstimator([_person(0.2, 0.2, 0.4)])
    frames = [TrackedFrame(i, _person(0.4, 0.4, 0.1), iou=0.9) for i in range(5)]

    refine_landmarks(frames, reader, estimator, margin=0.5)

    assert estimator.sizes, "nothing was refined"
    for height, width in estimator.sizes:
        assert height < 480 and width < 640


def test_refined_landmarks_come_back_in_frame_coordinates(reader):
    # The stub reports the person filling its crop; mapped back, that must land
    # inside the crop's place in the frame, not at 0-1 of the whole frame.
    estimator = CropRecordingEstimator([_person(0.0, 0.0, 1.0)])
    original = _person(0.4, 0.4, 0.1)
    result = refine_landmarks([TrackedFrame(0, original, iou=0.9)], reader, estimator, margin=0.5)

    box = result[0].person.bounding_box
    assert 0.0 <= box.x <= 1.0 and 0.0 <= box.y <= 1.0
    assert box.width < 1.0, "a crop's full width must map to less than the frame"


def test_gaps_are_left_alone(reader):
    estimator = CropRecordingEstimator([_person(0.2, 0.2, 0.4)])
    frames = [
        TrackedFrame(0, _person(0.4, 0.4, 0.1), iou=0.9),
        TrackedFrame(1, None),
        TrackedFrame(2, _person(0.4, 0.4, 0.1), iou=0.8),
    ]

    result = refine_landmarks(frames, reader, estimator, margin=0.5)

    assert result[1].is_gap
    assert estimator.calls == 2, "a gap has no box to crop around"


def test_a_frame_the_crop_finds_nobody_in_keeps_its_first_pass_answer(reader):
    """A failed crop is a worse answer than a coarse one, not an absence."""
    estimator = CropRecordingEstimator([])          # finds nobody
    original = _person(0.4, 0.4, 0.1)

    result = refine_landmarks([TrackedFrame(0, original, iou=0.9)], reader, estimator, margin=0.5)

    assert result[0].person == original
    assert not result[0].is_gap


def test_match_confidence_is_carried_through(reader):
    estimator = CropRecordingEstimator([_person(0.2, 0.2, 0.4)])
    result = refine_landmarks(
        [TrackedFrame(0, _person(0.4, 0.4, 0.1), iou=0.77)], reader, estimator, margin=0.5
    )
    assert result[0].iou == pytest.approx(0.77)


def test_a_tiny_box_is_not_worth_cropping(reader):
    estimator = CropRecordingEstimator([_person(0.2, 0.2, 0.4)])
    frames = [TrackedFrame(0, _person(0.5, 0.5, 0.001), iou=0.9)]

    result = refine_landmarks(frames, reader, estimator, margin=0.0)

    assert estimator.calls == 0
    assert result[0].person is not None


def test_output_stays_in_frame_order(reader):
    estimator = CropRecordingEstimator([_person(0.2, 0.2, 0.4)])
    frames = [TrackedFrame(i, _person(0.4, 0.4, 0.1), iou=0.9) for i in range(6)]

    result = refine_landmarks(frames, reader, estimator, margin=0.5)

    assert [f.frame_index for f in result] == list(range(6))


def test_a_refinement_that_relocates_the_person_is_rejected(reader):
    """Sharpening the landmarks is the job; moving the climber is a failure.

    A small crop can contain a different reading of the scene, and taking it
    would put the skeleton somewhere the first pass never saw anybody.
    """
    original = _person(0.4, 0.4, 0.1)
    # The stub reports someone in the far corner of every crop.
    estimator = CropRecordingEstimator([_person(0.9, 0.9, 0.05)])

    result = refine_landmarks(
        [TrackedFrame(0, original, iou=0.9)], reader, estimator, margin=0.5
    )

    assert result[0].person == original


def test_a_refinement_that_agrees_is_accepted(reader):
    original = _person(0.4, 0.4, 0.2)
    # Same region of the crop, so it maps back over the original.
    estimator = CropRecordingEstimator([_person(0.25, 0.25, 0.5)])

    result = refine_landmarks(
        [TrackedFrame(0, original, iou=0.9)], reader, estimator, margin=0.5
    )

    assert result[0].person != original, "an agreeing refinement should be taken"
