"""Job and keypoint endpoints over HTTP."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.db import (
    SqlAlchemyAnalysisRunRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyVideoRepository,
    session_scope,
)
from app.db.repository import JobKind
from app.ingest.job import IngestJobHandler
from app.jobs.base import JobQueue
from app.jobs.runtime import set_queue
from app.keypoints import KeypointMetadata, ParquetKeypointStore
from app.main import create_app
from app.pose.base import Landmark, PersonPose
from app.tracking import TrackedFrame

from tests.conftest import InlineJobQueue
from tests.test_candidates import FakeEstimator, _person



class SplitQueue(JobQueue):
    """Ingest runs inline; analysis is only recorded.

    These tests are about the API's handling of jobs, not about analysis - but
    a video still has to be normalised before it can be analysed at all.
    """

    def __init__(self, settings):
        self._inline = InlineJobQueue(settings, {JobKind.INGEST: IngestJobHandler(settings)})
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        from app.db import session_scope
        from app.db.sqlalchemy_repository import SqlAlchemyJobRepository

        with session_scope() as session:
            job = SqlAlchemyJobRepository(session).get(job_id)
        if job is not None and job.kind is JobKind.INGEST:
            self._inline.enqueue(job_id)
            return
        self.enqueued.append(job_id)

    def start(self) -> None:
        pass

    def stop(self, timeout: float | None = None) -> None:
        pass

    @property
    def depth(self) -> int:
        return len(self.enqueued)


@pytest.fixture
def queue(settings):
    split = SplitQueue(settings)
    set_queue(split)
    yield split
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
    assert job["analysis_run_id"]
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
        SqlAlchemyVideoRepository(session).set_latest_keypoints_path(video_id, path)

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


def _submit_run(client, video_id):
    response = client.post(f"/videos/{video_id}/analyse", json={"candidate_index": 0})
    assert response.status_code == 202
    return response.json()


def test_analysis_runs_are_listed_newest_first(client, video_id):
    submitted = [_submit_run(client, video_id) for _ in range(3)]

    response = client.get(f"/videos/{video_id}/analysis-runs")

    assert response.status_code == 200
    runs = response.json()
    assert runs == sorted(
        runs, key=lambda run: (run["created_at"], run["id"]), reverse=True
    )
    assert {run["id"] for run in runs} == {job["analysis_run_id"] for job in submitted}
    assert {run["video_id"] for run in runs} == {video_id}
    assert {run["execution"]["job_id"] for run in runs} == {job["id"] for job in submitted}


def test_analysis_run_detail_exposes_provenance_and_execution(client, settings, video_id):
    job = _submit_run(client, video_id)

    response = client.get(f"/videos/{video_id}/analysis-runs/{job['analysis_run_id']}")

    assert response.status_code == 200
    run = response.json()
    assert run["id"] == job["analysis_run_id"]
    assert run["video_id"] == video_id
    assert run["seed_frame_index"] >= 0
    assert run["selected_candidate_index"] == 0
    assert set(run["seed_bounding_box"]) == {"x", "y", "width", "height"}
    assert run["pose_configuration"] == {
        "model": settings.pose_model,
        "max_people": settings.max_people,
    }
    assert run["tracking_configuration"]["min_iou"] == pytest.approx(0.3)
    assert "max_gap_frames" in run["tracking_configuration"]
    assert run["created_at"]
    assert run["result_available"] is False
    assert run["keypoints_url"] is None
    assert run["execution"]["status"] == "queued"
    assert run["execution"]["job_id"] == job["id"]


def test_run_history_presents_completed_running_and_failed_jobs(client, video_id):
    completed = _submit_run(client, video_id)
    running = _submit_run(client, video_id)
    failed = _submit_run(client, video_id)
    with session_scope() as session:
        jobs = SqlAlchemyJobRepository(session)
        runs = SqlAlchemyAnalysisRunRepository(session)
        jobs.mark_done(completed["id"])
        runs.set_keypoints_path(completed["analysis_run_id"], "keypoints/completed.parquet")
        jobs.mark_running(running["id"])
        jobs.mark_failed(failed["id"], "pose estimator stopped")

    history = client.get(f"/videos/{video_id}/analysis-runs").json()
    by_id = {run["id"]: run for run in history}

    assert by_id[completed["analysis_run_id"]]["execution"]["status"] == "done"
    assert by_id[completed["analysis_run_id"]]["result_available"] is True
    assert by_id[running["analysis_run_id"]]["execution"]["status"] == "running"
    assert by_id[running["analysis_run_id"]]["execution"]["progress"] == 0.0
    assert by_id[failed["analysis_run_id"]]["execution"]["status"] == "failed"
    assert by_id[failed["analysis_run_id"]]["execution"]["error"] == "pose estimator stopped"


def test_run_is_scoped_to_its_video(client, make_video, video_id):
    source = make_video(name="other.mp4", seconds=1.0)
    with source.open("rb") as handle:
        other_id = client.post(
            "/videos", files={"file": ("other.mp4", handle, "video/mp4")}
        ).json()["id"]
    client.get(f"/videos/{other_id}/candidates")
    other_run = _submit_run(client, other_id)

    detail = client.get(
        f"/videos/{video_id}/analysis-runs/{other_run['analysis_run_id']}"
    )
    keypoints = client.get(
        f"/videos/{video_id}/keypoints?analysis_run_id={other_run['analysis_run_id']}"
    )

    assert detail.status_code == 404
    assert keypoints.status_code == 404


def test_missing_run_is_404(client, video_id):
    assert client.get(f"/videos/{video_id}/analysis-runs/missing").status_code == 404
    assert client.get(f"/videos/{video_id}/keypoints?analysis_run_id=missing").status_code == 404


def test_run_specific_keypoints_are_independent_and_latest_stays_default(
    client, settings, video_id
):
    first = _submit_run(client, video_id)
    second = _submit_run(client, video_id)
    first_path = _publish_result(settings, video_id, first, x=0.2)
    second_path = _publish_result(settings, video_id, second, x=0.8, make_latest=True)

    first_result = client.get(
        f"/videos/{video_id}/keypoints?analysis_run_id={first['analysis_run_id']}"
    ).json()
    second_result = client.get(
        f"/videos/{video_id}/keypoints?analysis_run_id={second['analysis_run_id']}"
    ).json()
    latest_result = client.get(f"/videos/{video_id}/keypoints").json()

    assert first_path != second_path
    assert first_result["frames"][0][0][0] == pytest.approx(0.2)
    assert second_result["frames"][0][0][0] == pytest.approx(0.8)
    assert latest_result == second_result
    assert latest_result["analysis_run_id"] == second["analysis_run_id"]


def test_unfinished_run_has_no_keypoint_result(client, video_id):
    job = _submit_run(client, video_id)

    response = client.get(
        f"/videos/{video_id}/keypoints?analysis_run_id={job['analysis_run_id']}"
    )

    assert response.status_code == 409
    assert "does not have a completed result" in response.json()["detail"]


def _publish_result(settings, video_id, job, x, make_latest=False):
    person = PersonPose(
        landmarks=(Landmark(x=x, y=0.5, z=0.0, visibility=0.9, presence=0.9),)
    )
    path = ParquetKeypointStore(settings).write(
        KeypointMetadata(
            video_id=video_id,
            fps=30.0,
            frame_count=1,
            landmark_names=("nose",),
            landmark_connections=(),
            pose_model="lite",
            min_iou=0.3,
            analysis_run_id=job["analysis_run_id"],
        ),
        [TrackedFrame(0, person, iou=1.0)],
    )
    with session_scope() as session:
        SqlAlchemyAnalysisRunRepository(session).set_keypoints_path(job["analysis_run_id"], path)
        SqlAlchemyJobRepository(session).mark_done(job["id"])
        if make_latest:
            SqlAlchemyVideoRepository(session).set_latest_keypoints_path(video_id, path)
    return path
