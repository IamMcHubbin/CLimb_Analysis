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

    def square_expanded(self, margin: float) -> "BoundingBox":
        """A square box around the same centre, grown by ``margin``, clamped.

        Square because the pose model works on a square crop and will letterbox
        anything else, wasting the pixels the crop exists to provide. Clamped
        to the frame, which can make the result non-square again at an edge -
        still better than sampling outside the image.
        """
        half = max(self.width, self.height) * (1.0 + margin) / 2.0
        cx, cy = self.x + self.width / 2.0, self.y + self.height / 2.0
        x1, y1 = max(0.0, cx - half), max(0.0, cy - half)
        x2, y2 = min(1.0, cx + half), min(1.0, cy + half)
        return BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

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
