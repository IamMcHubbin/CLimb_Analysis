"""The repository contract. Written against the interface, not SQLAlchemy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db import SqlAlchemyVideoRepository, session_scope
from app.db.repository import VideoRecord, VideoStatus


def _record(video_id: str = "abc123") -> VideoRecord:
    return VideoRecord(
        id=video_id,
        original_filename="IMG_0001.MOV",
        created_at=datetime.now(timezone.utc),
        status=VideoStatus.READY,
        stored_path=f"videos/{video_id}/normalised.mp4",
        width=1280,
        height=720,
        fps=30.0,
        frame_count=900,
        duration_seconds=30.0,
        size_bytes=1234567,
        source_width=1080,
        source_height=1920,
        source_fps=29.97,
        source_rotation=90,
        source_codec="h264",
        source_variable_frame_rate=True,
    )


@pytest.fixture
def repository(settings):
    with session_scope() as session:
        yield SqlAlchemyVideoRepository(session)


def test_round_trips_a_record(repository):
    stored = repository.add(_record())
    fetched = repository.get(stored.id)
    assert fetched == stored
    assert fetched.source_variable_frame_rate is True
    assert fetched.source_rotation == 90


def test_missing_video_is_none(repository):
    assert repository.get("does-not-exist") is None


def test_lists_newest_first(repository):
    older = _record("older")
    newer = _record("newer")
    repository.add(older)
    repository.add(
        VideoRecord(**{**newer.__dict__, "created_at": datetime.now(timezone.utc)})
    )
    listed = repository.list()
    assert [record.id for record in listed] == ["newer", "older"]


def test_pagination(repository):
    for index in range(5):
        repository.add(_record(f"video{index}"))
    assert len(repository.list(limit=2)) == 2
    assert len(repository.list(limit=2, offset=4)) == 1


def test_sets_keypoints_path(repository):
    stored = repository.add(_record())
    assert stored.keypoints_path is None
    repository.set_latest_keypoints_path(stored.id, "keypoints/abc123.parquet")
    assert repository.get(stored.id).keypoints_path == "keypoints/abc123.parquet"


def test_set_latest_keypoints_path_on_unknown_video(repository):
    with pytest.raises(KeyError):
        repository.set_latest_keypoints_path("nope", "keypoints/nope.parquet")


def test_delete(repository):
    repository.add(_record())
    assert repository.delete("abc123") is True
    assert repository.get("abc123") is None
    assert repository.delete("abc123") is False
