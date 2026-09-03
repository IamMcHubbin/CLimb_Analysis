"""IoU tracking, and specifically its refusal to guess."""

from __future__ import annotations

import pytest

from app.geometry import BoundingBox
from app.pose.base import Landmark, PersonPose
from app.tracking import IouTracker, TrackingConfig


def _person(x: float, y: float, size: float = 0.2) -> PersonPose:
    corners = [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]
    return PersonPose(
        landmarks=tuple(
            Landmark(x=cx, y=cy, z=0.0, visibility=0.9, presence=0.9) for cx, cy in corners
        )
    )


def _box(x: float, y: float, size: float = 0.2) -> BoundingBox:
    return BoundingBox(x, y, size, size)


def test_follows_a_person_who_moves_gradually():
    tracker = IouTracker(_box(0.1, 0.1))

    results = [
        tracker.update(index, (_person(0.1 + index * 0.02, 0.1),))
        for index in range(5)
    ]

    assert all(not result.is_gap for result in results)
    assert tracker.reference_box.x == 0.1 + 4 * 0.02


def test_picks_the_best_overlap_when_several_people_are_present():
    tracker = IouTracker(_box(0.1, 0.1))

    # The tracked person barely moved; a second person is elsewhere.
    result = tracker.update(0, (_person(0.7, 0.7), _person(0.11, 0.1)))

    assert not result.is_gap
    assert result.person.bounding_box.x == 0.11


def test_a_poor_match_is_a_gap_not_a_snap():
    """The whole point: an unmatched frame is a hole, not the nearest person."""
    tracker = IouTracker(_box(0.1, 0.1), TrackingConfig(min_iou=0.3))

    # Someone else entirely, on the other side of the frame.
    result = tracker.update(0, (_person(0.8, 0.8),))

    assert result.is_gap
    assert result.person is None
    # The reference has not drifted towards the rejected detection.
    assert tracker.reference_box == _box(0.1, 0.1)


def test_no_detections_at_all_is_a_gap():
    tracker = IouTracker(_box(0.1, 0.1))
    assert tracker.update(0, ()).is_gap


def test_track_is_recovered_after_a_brief_occlusion():
    tracker = IouTracker(_box(0.1, 0.1))

    tracker.update(0, (_person(0.1, 0.1),))
    assert tracker.update(1, ()).is_gap
    assert tracker.update(2, ()).is_gap
    # Reappears roughly where it left off.
    recovered = tracker.update(3, (_person(0.12, 0.11),))

    assert not recovered.is_gap
    assert recovered.iou > 0.3


def test_gap_limit_gives_up_permanently():
    tracker = IouTracker(_box(0.1, 0.1), TrackingConfig(max_gap_frames=2))

    for index in range(3):
        assert tracker.update(index, ()).is_gap

    assert tracker.is_lost
    # Even an exact match is refused once the track is abandoned: after that
    # long unseen, an overlapping box is not evidence of the same person.
    assert tracker.update(3, (_person(0.1, 0.1),)).is_gap


def test_unlimited_gaps_by_default():
    tracker = IouTracker(_box(0.1, 0.1))
    for index in range(500):
        tracker.update(index, ())
    assert not tracker.is_lost
    assert not tracker.update(500, (_person(0.1, 0.1),)).is_gap


def test_reported_iou_reflects_match_quality():
    tracker = IouTracker(_box(0.1, 0.1))

    exact = tracker.update(0, (_person(0.1, 0.1),))
    drifted = tracker.update(1, (_person(0.15, 0.1),))

    assert exact.iou == pytest.approx(1.0)
    assert 0.3 < drifted.iou < 1.0
    assert tracker.update(2, ()).iou == 0.0


def test_iou_never_exceeds_one():
    box = BoundingBox(0.1, 0.1, 0.2, 0.2)
    assert box.iou(BoundingBox(0.1, 0.1, 0.2, 0.2)) <= 1.0
    assert box.iou(BoundingBox(0.9, 0.9, 0.05, 0.05)) == 0.0
