"""The candidate endpoints, over HTTP."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.db.repository import JobKind
from app.ingest.job import IngestJobHandler
from app.main import create_app

from tests.conftest import InlineJobQueue

from tests.test_candidates import FakeEstimator, _person


@pytest.fixture
def client(settings):
    """A client whose candidate detection uses a stub, not a pose model."""
    estimator = FakeEstimator([_person(0.1, 0.1, 0.3), _person(0.6, 0.2, 0.15)])

    @contextmanager
    def factory():
        yield estimator

    from app.jobs.runtime import set_queue

    inline = InlineJobQueue(settings, {JobKind.INGEST: IngestJobHandler(settings)})
    set_queue(inline)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings
    app.dependency_overrides[deps.get_estimator_factory] = lambda: factory
    app.dependency_overrides[deps.get_job_queue] = lambda: inline
    with TestClient(app) as test_client:
        yield test_client
    set_queue(None)


@pytest.fixture
def video_id(client, make_video):
    source = make_video(seconds=2.0)
    with source.open("rb") as handle:
        return client.post("/videos", files={"file": ("clip.mp4", handle, "video/mp4")}).json()["id"]


def test_returns_candidates_with_a_frame_url(client, video_id):
    response = client.get(f"/videos/{video_id}/candidates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["frame_index"] == 30  # 60 frames, middle of the clip
    assert payload["timestamp_seconds"] == pytest.approx(1.0)
    assert payload["frame_url"] == f"/videos/{video_id}/candidates/frame.jpg"
    assert (payload["frame_width"], payload["frame_height"]) == (320, 240)

    assert [candidate["index"] for candidate in payload["candidates"]] == [0, 1]
    for candidate in payload["candidates"]:
        box = candidate["bounding_box"]
        # Normalised, so the client can scale them to any rendered size.
        assert 0.0 <= box["x"] <= 1.0
        assert 0.0 <= box["y"] <= 1.0
        assert 0.0 < box["width"] <= 1.0


def test_frame_image_is_served(client, video_id):
    client.get(f"/videos/{video_id}/candidates")

    response = client.get(f"/videos/{video_id}/candidates/frame.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_frame_image_before_detection_is_404(client, video_id):
    response = client.get(f"/videos/{video_id}/candidates/frame.jpg")
    assert response.status_code == 404


def test_explicit_frame_index(client, video_id):
    payload = client.get(f"/videos/{video_id}/candidates", params={"frame_index": 5}).json()
    assert payload["frame_index"] == 5


def test_unknown_video(client):
    assert client.get("/videos/nope/candidates").status_code == 404
    assert client.get("/videos/nope/candidates/frame.jpg").status_code == 404
