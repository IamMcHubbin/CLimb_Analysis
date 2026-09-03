"""Keypoint storage round trips, including how gaps are represented."""

from __future__ import annotations

import pytest

from app.keypoints import KeypointMetadata, ParquetKeypointStore
from app.pose.base import Landmark, PersonPose
from app.tracking import TrackedFrame


def _metadata(frame_count: int = 5) -> KeypointMetadata:
    return KeypointMetadata(
        video_id="vid123",
        fps=30.0,
        frame_count=frame_count,
        landmark_names=("nose", "left_shoulder", "right_shoulder"),
        landmark_connections=((1, 2), (0, 1)),
        pose_model="lite",
        min_iou=0.3,
    )


def _person(offset: float) -> PersonPose:
    return PersonPose(
        landmarks=tuple(
            Landmark(x=offset + i * 0.01, y=0.5, z=0.0, visibility=0.8, presence=0.9)
            for i in range(3)
        )
    )


def test_round_trips_a_track(settings):
    store = ParquetKeypointStore(settings)
    frames = [TrackedFrame(index, _person(0.1 * index), iou=0.9) for index in range(5)]

    path = store.write(_metadata(), frames)
    data = store.read(path)

    assert data.metadata.landmark_names == ("nose", "left_shoulder", "right_shoulder")
    assert data.metadata.landmark_connections == ((1, 2), (0, 1))
    assert set(data.frames) == {0, 1, 2, 3, 4}
    assert len(data.frames[0]) == 3
    assert data.frames[2][0].x == pytest.approx(0.2, abs=1e-6)
    assert data.ious[3] == pytest.approx(0.9)


def test_gaps_are_absent_rows_not_placeholders(settings):
    store = ParquetKeypointStore(settings)
    frames = [
        TrackedFrame(0, _person(0.1), iou=0.9),
        TrackedFrame(1, None),
        TrackedFrame(2, None),
        TrackedFrame(3, _person(0.4), iou=0.8),
        TrackedFrame(4, None),
    ]

    data = store.read(store.write(_metadata(), frames))

    # Nothing is interpolated across the hole; the frames simply are not there.
    assert set(data.frames) == {0, 3}
    assert data.tracked_frame_count == 2
    assert data.gap_frame_count == 3


def test_a_track_with_no_matches_at_all(settings):
    store = ParquetKeypointStore(settings)
    frames = [TrackedFrame(index, None) for index in range(5)]

    data = store.read(store.write(_metadata(), frames))

    assert data.frames == {}
    assert data.gap_frame_count == 5


def test_batching_does_not_change_the_result(settings):
    frames = [TrackedFrame(index, _person(0.01 * index), iou=0.7) for index in range(50)]
    unbatched = ParquetKeypointStore(settings, batch_frames=1000)
    batched = ParquetKeypointStore(settings, batch_frames=7)

    first = unbatched.read(unbatched.write(_metadata(50), frames))
    second = batched.read(batched.write(_metadata(50), frames))

    assert first.frames.keys() == second.frames.keys()
    assert first.frames[49] == second.frames[49]


def test_path_is_stored_relative_to_the_data_directory(settings):
    store = ParquetKeypointStore(settings)
    path = store.write(_metadata(), [TrackedFrame(0, _person(0.1), iou=1.0)])

    assert not path.startswith("/")
    assert settings.resolve(path).exists()

    store.delete(path)
    assert not settings.resolve(path).exists()
