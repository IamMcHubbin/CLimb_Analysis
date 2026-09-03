"""Storing a finished track.

One file per video, written once by the analysis worker and read back whole by
the API. Coordinates are normalised 0-1, so the data stays valid if the video
is ever re-encoded at another size.

Frames are stored long-format - one row per landmark per frame - rather than a
column per joint, because the pose model is expected to change and a different
model has a different number of joints. A frame that is absent from the file is
a gap: the tracker saw nobody it could match. Every frame of the clip is
processed, so absence is unambiguous.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.config import Settings, settings as default_settings
from app.pose.base import Landmark
from app.tracking import TrackedFrame

METADATA_KEY = b"climb_analysis"

_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int32()),
        pa.field("landmark_index", pa.int16()),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("z", pa.float32()),
        pa.field("visibility", pa.float32()),
        pa.field("presence", pa.float32()),
        # The tracker's confidence in this frame's match, repeated per row.
        pa.field("iou", pa.float32()),
    ]
)


@dataclass(frozen=True)
class KeypointMetadata:
    """Everything needed to interpret the rows, and to know how they were made."""

    video_id: str
    fps: float
    frame_count: int
    landmark_names: tuple[str, ...]
    landmark_connections: tuple[tuple[int, int], ...]
    pose_model: str
    min_iou: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "video_id": self.video_id,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "landmark_names": list(self.landmark_names),
                "landmark_connections": [list(pair) for pair in self.landmark_connections],
                "pose_model": self.pose_model,
                "min_iou": self.min_iou,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "KeypointMetadata":
        data = json.loads(payload)
        return cls(
            video_id=data["video_id"],
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            landmark_names=tuple(data["landmark_names"]),
            landmark_connections=tuple((int(a), int(b)) for a, b in data["landmark_connections"]),
            pose_model=data["pose_model"],
            min_iou=float(data["min_iou"]),
        )


@dataclass(frozen=True)
class KeypointData:
    """A track read back. ``frames`` is sparse - a missing index is a gap."""

    metadata: KeypointMetadata
    frames: dict[int, tuple[Landmark, ...]]
    ious: dict[int, float]

    @property
    def tracked_frame_count(self) -> int:
        return len(self.frames)

    @property
    def gap_frame_count(self) -> int:
        return self.metadata.frame_count - len(self.frames)


class KeypointStore(abc.ABC):
    """Where finished tracks live."""

    @abc.abstractmethod
    def write(
        self,
        metadata: KeypointMetadata,
        tracked_frames: Iterable[TrackedFrame],
    ) -> str:
        """Persist a track. Returns the path to store on the video row."""

    @abc.abstractmethod
    def read(self, relative_path: str) -> KeypointData:
        """Read a track back."""

    @abc.abstractmethod
    def delete(self, relative_path: str) -> None:
        """Remove a stored track."""


class ParquetKeypointStore(KeypointStore):
    def __init__(self, settings: Settings = default_settings, batch_frames: int = 300) -> None:
        self._settings = settings
        # Row groups are flushed every this many frames so peak memory does not
        # scale with clip length.
        self._batch_frames = batch_frames

    def path_for(self, video_id: str) -> Path:
        return self._settings.keypoints_dir / f"{video_id}.parquet"

    def write(self, metadata: KeypointMetadata, tracked_frames: Iterable[TrackedFrame]) -> str:
        destination = self.path_for(metadata.video_id)
        destination.parent.mkdir(parents=True, exist_ok=True)

        schema = _SCHEMA.with_metadata({METADATA_KEY: metadata.to_json().encode()})
        writer = pq.ParquetWriter(destination, schema, compression="snappy")
        try:
            for batch in _batched_record_batches(tracked_frames, schema, self._batch_frames):
                writer.write_batch(batch)
        except BaseException:
            writer.close()
            destination.unlink(missing_ok=True)
            raise
        writer.close()
        return self._settings.relative(destination)

    def read(self, relative_path: str) -> KeypointData:
        path = self._settings.resolve(relative_path)
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path.name} is missing its keypoint metadata")
        metadata = KeypointMetadata.from_json(raw.decode())

        frames: dict[int, list[Landmark]] = {}
        ious: dict[int, float] = {}
        columns = table.to_pydict()
        for index in range(table.num_rows):
            frame_index = columns["frame_index"][index]
            frames.setdefault(frame_index, []).append(
                Landmark(
                    x=columns["x"][index],
                    y=columns["y"][index],
                    z=columns["z"][index],
                    visibility=columns["visibility"][index],
                    presence=columns["presence"][index],
                )
            )
            ious[frame_index] = columns["iou"][index]

        return KeypointData(
            metadata=metadata,
            frames={index: tuple(landmarks) for index, landmarks in frames.items()},
            ious=ious,
        )

    def delete(self, relative_path: str) -> None:
        self._settings.resolve(relative_path).unlink(missing_ok=True)


def _batched_record_batches(
    tracked_frames: Iterable[TrackedFrame],
    schema: pa.Schema,
    batch_frames: int,
) -> Iterator[pa.RecordBatch]:
    """Turn tracked frames into Arrow record batches, skipping gaps."""
    columns: dict[str, list] = {name: [] for name in schema.names}
    frames_in_batch = 0

    def flush() -> pa.RecordBatch | None:
        if not columns["frame_index"]:
            return None
        batch = pa.record_batch([pa.array(columns[name], type=schema.field(name).type)
                                 for name in schema.names], schema=schema)
        for values in columns.values():
            values.clear()
        return batch

    for tracked in tracked_frames:
        if tracked.person is None:
            # Gaps are simply not written. Absence is the representation.
            continue
        for landmark_index, landmark in enumerate(tracked.person.landmarks):
            columns["frame_index"].append(tracked.frame_index)
            columns["landmark_index"].append(landmark_index)
            columns["x"].append(landmark.x)
            columns["y"].append(landmark.y)
            columns["z"].append(landmark.z)
            columns["visibility"].append(landmark.visibility)
            columns["presence"].append(landmark.presence)
            columns["iou"].append(tracked.iou)
        frames_in_batch += 1
        if frames_in_batch >= batch_frames:
            batch = flush()
            frames_in_batch = 0
            if batch is not None:
                yield batch

    final = flush()
    if final is not None:
        yield final
