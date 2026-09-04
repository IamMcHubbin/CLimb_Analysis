"""The ingest endpoint, exercised through HTTP."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.db.repository import JobKind
from app.ingest.job import IngestJobHandler
from app.jobs.runtime import set_queue
from app.main import create_app

from tests.conftest import InlineJobQueue


@pytest.fixture
def queue(settings):
    """Jobs run inline, on this thread, using the test's settings.

    The app's own queue is built from the module-level settings and would look
    for files in the real data directory.
    """
    inline = InlineJobQueue(settings, {JobKind.INGEST: IngestJobHandler(settings)})
    set_queue(inline)
    yield inline
    set_queue(None)


@pytest.fixture
def client(settings, queue):
    app = create_app()
    # The app-wide settings object is built at import time; point the request
    # dependencies at the throwaway directory for this test instead.
    app.dependency_overrides[deps.get_settings] = lambda: settings
    app.dependency_overrides[deps.get_job_queue] = lambda: queue
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_is_accepted_and_normalised(client, make_video):
    source = make_video(width=640, height=480, rotate=90, vfr=True)

    with source.open("rb") as handle:
        response = client.post("/videos", files={"file": ("IMG_0001.MOV", handle, "video/quicktime")})

    # 202, not 201: the bytes are in, the transcode has not run yet. Holding
    # the request open for it is what invites a proxy timeout.
    assert response.status_code == 202
    payload = client.get(f"/videos/{response.json()['id']}").json()
    assert payload["status"] == "ready"
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


def test_a_non_video_upload_fails_during_normalisation(client):
    """Accepted at the door, rejected by ffmpeg, reported on the video."""
    response = client.post("/videos", files={"file": ("notes.txt", b"not a video", "text/plain")})
    assert response.status_code == 202

    video = client.get(f"/videos/{response.json()['id']}").json()
    assert video["status"] == "failed"
    assert "video stream" in video["ingest_error"]
    # And the message says nothing about where the file lived on the server.
    assert "/" not in video["ingest_error"]


def test_endpoints_refuse_a_video_that_is_not_ready(client, settings, make_video):
    """With nothing draining the queue, the video stays pending."""
    # Override the injected queue, not the module-level one: the request path
    # uses the dependency.
    client.app.dependency_overrides[deps.get_job_queue] = lambda: InlineJobQueue(settings, {})
    source = make_video(seconds=1.0)
    with source.open("rb") as handle:
        video_id = client.post(
            "/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
        ).json()["id"]

    assert client.get(f"/videos/{video_id}").json()["status"] == "pending"
    assert client.get(f"/videos/{video_id}/candidates").status_code == 409
    assert client.get(f"/videos/{video_id}/file").status_code == 409


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


def test_the_upload_cap_is_fifty_megabytes(settings, make_video):
    """The default cap is 50 MB."""
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_footage_can_be_deleted_on_demand(client, make_video):
    source = make_video(seconds=1.0)
    with source.open("rb") as handle:
        video_id = client.post(
            "/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
        ).json()["id"]
    # Re-read after normalisation: the upload response is the pending record.
    video = client.get(f"/videos/{video_id}").json()
    assert video["has_footage"] is True
    assert client.get(f"/videos/{video_id}/file").status_code == 200

    deleted = client.request("DELETE", f"/videos/{video_id}/footage").json()

    assert deleted["has_footage"] is False
    assert deleted["footage_expires_at"] is None
    # The clip is gone; the row and its metadata are not.
    assert client.get(f"/videos/{video_id}").json()["frame_count"] == video["frame_count"]
    assert client.get(f"/videos/{video_id}/file").status_code == 410


def test_deleting_footage_twice_is_not_an_error(client, make_video):
    source = make_video(seconds=1.0)
    with source.open("rb") as handle:
        video_id = client.post(
            "/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
        ).json()["id"]

    assert client.request("DELETE", f"/videos/{video_id}/footage").status_code == 200
    assert client.request("DELETE", f"/videos/{video_id}/footage").status_code == 200


def test_an_unanalysed_video_reports_when_its_footage_expires(client, make_video):
    source = make_video(seconds=1.0)
    with source.open("rb") as handle:
        video = client.post("/videos", files={"file": ("clip.mp4", handle, "video/mp4")}).json()

    assert video["footage_expires_at"] is not None
    assert video["footage_expires_at"] > video["created_at"]
