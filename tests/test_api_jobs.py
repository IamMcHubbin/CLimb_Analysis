"""Job and keypoint endpoints over HTTP."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.jobs.runtime import set_queue
from app.main import create_app

from tests.test_candidates import FakeEstimator, _person
from tests.test_jobs import RecordingQueue


@pytest.fixture
def queue():
    """Swap the real worker out: these tests are about the API, not analysis."""
    recording = RecordingQueue()
    set_queue(recording)
    yield recording
    set_queue(None)


@pytest.fixture
def client(settings, queue):
    estimator = FakeEstimator([_person(0.1, 0.1, 0.3)])

    @contextmanager
    def factory():
        yield estimator

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings
    app.dependency_overrides[deps.get_estimator_factory] = lambda: factory
    app.dependency_overrides[deps.get_job_queue] = lambda: queue
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def video_id(client, make_video):
    source = make_video(seconds=1.0)
    with source.open("rb") as handle:
        video = client.post("/videos", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
    client.get(f"/videos/{video['id']}/candidates")
    return video["id"]


def test_analyse_returns_a_queued_job_immediately(client, video_id, queue):
    response = client.post(f"/videos/{video_id}/analyse", json={"candidate_index": 0})

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["progress"] == 0.0
    assert job["video_id"] == video_id
    assert queue.enqueued == [job["id"]]


def test_job_status_is_readable(client, video_id):
    job_id = client.post(f"/videos/{video_id}/analyse", json={"candidate_index": 0}).json()["id"]

    status = client.get(f"/jobs/{job_id}").json()

    assert status["id"] == job_id
    assert status["status"] == "queued"
    assert status["finished_at"] is None


def test_analysing_an_unknown_candidate_is_rejected(client, video_id):
    response = client.post(f"/videos/{video_id}/analyse", json={"candidate_index": 42})
    assert response.status_code == 422


def test_analysing_an_unknown_video_is_404(client):
    assert client.post("/videos/nope/analyse", json={"candidate_index": 0}).status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_keypoints_before_analysis_is_a_conflict(client, video_id):
    response = client.get(f"/videos/{video_id}/keypoints")
    assert response.status_code == 409
    assert "not been analysed" in response.json()["detail"]


def test_keypoints_are_index_aligned_with_the_video(client, settings, video_id):
    """Gaps come back as nulls in place, so the client can index straight in."""
    from app.db import SqlAlchemyVideoRepository, session_scope
    from app.keypoints import KeypointMetadata, ParquetKeypointStore
    from app.pose.base import Landmark, PersonPose
    from app.tracking import TrackedFrame

    person = PersonPose(
        landmarks=(Landmark(0.5, 0.5, 0.0, 0.9, 0.9), Landmark(0.6, 0.6, 0.0, 0.4, 0.9))
    )
    frames = [
        TrackedFrame(0, person, iou=0.95),
        TrackedFrame(1, None),
        TrackedFrame(2, person, iou=0.80),
    ]
    store = ParquetKeypointStore(settings)
    path = store.write(
        KeypointMetadata(
            video_id=video_id,
            fps=30.0,
            frame_count=3,
            landmark_names=("head", "foot"),
            landmark_connections=((0, 1),),
            pose_model="lite",
            min_iou=0.3,
        ),
        frames,
    )
    with session_scope() as session:
        SqlAlchemyVideoRepository(session).set_keypoints_path(video_id, path)

    payload = client.get(f"/videos/{video_id}/keypoints").json()

    assert payload["frame_count"] == 3
    assert len(payload["frames"]) == 3
    assert payload["frames"][1] is None
    assert payload["match_iou"][1] is None
    assert payload["frames"][0][0][:2] == [0.5, 0.5]
    assert payload["match_iou"][0] == pytest.approx(0.95, abs=1e-3)
    assert payload["landmark_connections"] == [[0, 1]]
    assert payload["tracked_frame_count"] == 2
    assert payload["gap_frame_count"] == 1
