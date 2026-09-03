"""Shared geometry.

Lives outside the pose package because tracking, persistence and the API all
speak in boxes, and none of them should have to import a pose model to do it.
All coordinates are normalised 0-1 against the frame.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        if union <= 0:
            return 0.0
        # Clamped: identical boxes can compute to fractionally over 1.0 in
        # floating point, and a ratio above 1 would be nonsense downstream.
        return min(1.0, intersection / union)
