"""Pose estimation interface.

The model behind this will be swapped, so everything downstream - tracking,
storage, the frontend - depends only on the types here. Coordinates are always
normalised to 0-1 against the frame, never pixels, so keypoints stay valid if a
video is re-encoded at a different size.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum

from app.geometry import BoundingBox


__all__ = [
    "BoundingBox",
    "FramePose",
    "Landmark",
    "PersonPose",
    "PoseEstimator",
    "RunningMode",
]


class RunningMode(str, Enum):
    """Whether the estimator may carry state between calls.

    IMAGE treats every frame independently - what the candidate picker needs.
    VIDEO lets the model track between frames and requires timestamps that
    increase monotonically.
    """

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class Landmark:
    """One joint. ``x``/``y`` are normalised to the frame; ``z`` is roughly in
    the same scale as ``x`` and is relative to the hips, not metric."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class PersonPose:
    """One detected person in one frame."""

    landmarks: tuple[Landmark, ...]

    @property
    def bounding_box(self) -> BoundingBox:
        """Box enclosing every landmark, clamped to the frame.

        Occluded joints are extrapolated by the model and can land outside the
        image, which is why this clamps rather than trusting the raw extent.
        """
        if not self.landmarks:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        xs = [min(1.0, max(0.0, lm.x)) for lm in self.landmarks]
        ys = [min(1.0, max(0.0, lm.y)) for lm in self.landmarks]
        x, y = min(xs), min(ys)
        return BoundingBox(x=x, y=y, width=max(xs) - x, height=max(ys) - y)

    @property
    def mean_visibility(self) -> float:
        if not self.landmarks:
            return 0.0
        return sum(lm.visibility for lm in self.landmarks) / len(self.landmarks)


@dataclass(frozen=True)
class FramePose:
    """Everything detected in one frame."""

    frame_index: int
    timestamp_ms: int
    people: tuple[PersonPose, ...]


class PoseEstimator(abc.ABC):
    """Detects people in single frames.

    Implementations are not thread safe: the analysis worker is single
    threaded and each request that needs a one-off detection builds its own.
    """

    @property
    @abc.abstractmethod
    def landmark_names(self) -> tuple[str, ...]:
        """Names in landmark order. Defines the schema of the stored keypoints."""

    @property
    @abc.abstractmethod
    def landmark_connections(self) -> tuple[tuple[int, int], ...]:
        """Skeleton edges as (start, end) index pairs, for drawing."""

    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_names)

    @property
    def uses_roi_hint(self) -> bool:
        """Whether ``detect`` accepts a ``roi`` telling it where to look.

        Only top-down estimators - the ones that find people first and then
        pose each one - can save anything by being told where the person is,
        because it lets them skip finding them. An estimator that says True
        here must accept ``roi`` as a keyword argument. Callers must not pass
        ``roi`` to one that says False, which is the default, so existing
        implementations keep working untouched.
        """
        return False

    @abc.abstractmethod
    def detect(self, frame_bgr, timestamp_ms: int = 0) -> tuple[PersonPose, ...]:
        """Detect every person in a BGR frame (the layout OpenCV decodes into).

        In VIDEO mode ``timestamp_ms`` must increase between calls.

        Implementations advertising ``uses_roi_hint`` also accept a keyword
        ``roi: BoundingBox | None``. Given one, they may pose only that region
        and skip their own search; given None they must search the whole frame
        as usual. A caller relying on this has to accept that the result is no
        longer an independent opinion about where the person is - see the
        re-anchoring in ``app.analysis``.
        """

    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
