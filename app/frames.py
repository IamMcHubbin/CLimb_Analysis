"""Reading frames out of a normalised video.

Both the candidate picker (one frame, random access) and the analysis worker
(every frame, in order) go through here, so assumptions about frame indexing
live in one place. Only ever pointed at normalised files: constant frame rate
means index and timestamp convert exactly, which is not true of the originals.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class FrameReadError(RuntimeError):
    pass


def timestamp_ms_for_frame(frame_index: int, fps: float) -> int:
    """Frame index to milliseconds. Exact only because the file is constant rate."""
    return int(round(1000 * frame_index / fps))


class FrameReader:
    """A cv2.VideoCapture with the sharp edges covered.

    Use as a context manager; the capture is released on exit.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise FrameReadError(f"could not open {path}")

    @property
    def frame_count(self) -> int:
        """What the container claims. Trust the database's count instead - this
        is only useful for a sanity check."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_at(self, frame_index: int) -> np.ndarray:
        """Seek to a frame and return it.

        Seeking is per-keyframe internally, so this is only accurate because
        normalisation re-encodes every file we point it at.
        """
        if frame_index < 0:
            raise FrameReadError(f"negative frame index {frame_index}")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise FrameReadError(f"no frame at index {frame_index} in {self._path.name}")
        return frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) from the current position to the end."""
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return
            yield index, frame
            index += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise FrameReadError("could not encode frame as JPEG")
    return buffer.tobytes()
