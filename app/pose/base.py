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


class RunningMode(str, Enum):
    """Whether the estimator may carry state between calls.

    IMAGE treats every frame independently - what the candidate picker needs.
    VIDEO lets the model track between frames and requires timestamps that
    increase monotonically.
    """

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class BoundingBox:
    """Normalised box. ``x``/``y`` are the top-left corner."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union. The basis of frame-to-frame tracking."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


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
    def num_landmarks(self) -> int:
        return len(self.landmark_names)

    @abc.abstractmethod
    def detect(self, frame_bgr, timestamp_ms: int = 0) -> tuple[PersonPose, ...]:
        """Detect every person in a BGR frame (the layout OpenCV decodes into).

        In VIDEO mode ``timestamp_ms`` must increase between calls.
        """

    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
