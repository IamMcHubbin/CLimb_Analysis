"""Following one person across frames.

Multi-person detection is on, so every frame comes back with whoever the model
found, in no particular order and with no identity attached. This turns that
into one person's track by matching each frame's detections against where the
tracked person was last seen.

The rule that matters: when nothing matches well enough, the frame is a gap.
It is never filled with the best of a bad set. On a climbing wall the wrong
person is often the *closest* person - a spotter directly below, a queue at the
base - so snapping to the nearest box would produce a track that looks
plausible and is wrong, which is worse than a hole.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from app.geometry import BoundingBox
from app.pose.base import PersonPose


@dataclass(frozen=True)
class TrackingConfig:
    """Thresholds. Defaults are a starting point, to be tuned on real footage."""

    # Below this IoU against the last known position, a detection is not
    # considered the same person.
    min_iou: float = 0.3

    # Give up permanently after this many consecutive unmatched frames. None
    # keeps trying, which allows re-acquisition after an occlusion but also
    # risks latching onto whoever later occupies that part of the frame.
    max_gap_frames: int | None = None


@dataclass(frozen=True)
class TrackedFrame:
    """One frame of a track. ``person`` is None for a gap."""

    frame_index: int
    person: PersonPose | None
    iou: float = 0.0

    @property
    def is_gap(self) -> bool:
        return self.person is None


class Tracker(abc.ABC):
    """Turns per-frame detections into one person's track."""

    @abc.abstractmethod
    def update(self, frame_index: int, people: tuple[PersonPose, ...]) -> TrackedFrame:
        """Match this frame's detections against the tracked person."""


class IouTracker(Tracker):
    """Greedy IoU matching against the last confirmed position.

    The reference box is where the person was last *matched*, not where they
    were last predicted, so a brief occlusion can be recovered from: the track
    picks up again when someone reappears overlapping where it left off. There
    is no motion model, deliberately - this is meant to show what raw
    frame-to-frame association gives before anything smooths over its failures.
    """

    def __init__(self, seed_box: BoundingBox, config: TrackingConfig | None = None) -> None:
        self._reference = seed_box
        self._config = config or TrackingConfig()
        self._consecutive_gaps = 0
        self._lost = False

    @property
    def reference_box(self) -> BoundingBox:
        """Where the tracker currently believes the person is."""
        return self._reference

    @property
    def is_lost(self) -> bool:
        return self._lost

    def update(self, frame_index: int, people: tuple[PersonPose, ...]) -> TrackedFrame:
        if self._lost:
            return TrackedFrame(frame_index=frame_index, person=None)

        best_person: PersonPose | None = None
        best_iou = 0.0
        for person in people:
            iou = self._reference.iou(person.bounding_box)
            if iou > best_iou:
                best_person, best_iou = person, iou

        if best_person is None or best_iou < self._config.min_iou:
            self._consecutive_gaps += 1
            if (
                self._config.max_gap_frames is not None
                and self._consecutive_gaps > self._config.max_gap_frames
            ):
                self._lost = True
            # The reference is left where it was: the person may yet reappear
            # there, and moving it towards a rejected detection would drag the
            # track onto whoever that was.
            return TrackedFrame(frame_index=frame_index, person=None)

        self._consecutive_gaps = 0
        self._reference = best_person.bounding_box
        return TrackedFrame(frame_index=frame_index, person=best_person, iou=best_iou)
