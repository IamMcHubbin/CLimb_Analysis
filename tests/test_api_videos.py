"""The ingest endpoint, exercised through HTTP."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import create_app


@pytest.fixture
def client(settings):
    app = create_app()
    # The app-wide settings object is built at import time; point the request
    # dependencies at the throwaway directory for this test instead.
    app.dependency_overrides[deps.get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_returns_normalised_metadata(client, make_video):
    source = make_video(width=640, height=480, rotate=90, vfr=True)

    with source.open("rb") as handle:
        response = client.post("/videos", files={"file": ("IMG_0001.MOV", handle, "video/quicktime")})

    assert response.status_code == 201
    payload = response.json()
    # Rotation was baked in, so the stored file is portrait.
    assert (payload["width"], payload["height"]) == (480, 640)
    assert payload["fps"] == 30.0
    assert payload["frame_count"] > 0
    assert payload["has_keypoints"] is False
    # The provenance of the upload is recorded, not discarded.
    assert payload["source"]["rotation"] != 0
    assert payload["source"]["variable_frame_rate"] is True
    assert payload["original_filename"] == "IMG_0001.MOV"


def test_uploaded_video_can_be_fetched_back(client, make_video):
    source = make_video()
    with source.open("rb") as handle:
        video_id = client.post("/videos", files={"file": ("clip.mp4", handle, "video/mp4")}).json()["id"]

    assert client.get(f"/videos/{video_id}").json()["id"] == video_id
    assert client.get(f"/videos/{video_id}/file").status_code == 200
    assert [video["id"] for video in client.get("/videos").json()] == [video_id]


def test_upload_of_a_non_video_is_rejected(client):
    response = client.post("/videos", files={"file": ("notes.txt", b"not a video", "text/plain")})
    assert response.status_code == 415


def test_empty_upload_is_rejected(client):
    response = client.post("/videos", files={"file": ("empty.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_oversized_upload_is_rejected(settings, make_video):
    import dataclasses

    tiny_limit = dataclasses.replace(settings, max_upload_bytes=1024)
    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: tiny_limit
    source = make_video(seconds=2.0)
    # Entered as a context manager so the lifespan runs and the database exists.
    with TestClient(app) as client, source.open("rb") as handle:
        response = client.post("/videos", files={"file": ("big.mp4", handle, "video/mp4")})
    assert response.status_code == 413


def test_unknown_video_is_404(client):
    assert client.get("/videos/missing").status_code == 404
    assert client.get("/videos/missing/file").status_code == 404
