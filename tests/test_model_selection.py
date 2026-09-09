"""Model choice is request-local and bound to an exact detection response."""
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.analysis import AnalysisJobHandler
from app.db import session_scope, SqlAlchemyVideoRepository, SqlAlchemyAnalysisRunRepository
from app.db.sqlalchemy_repository import _decode_candidates
from app.main import create_app
from app.pose.base import Landmark, PersonPose, PoseEstimator
from app.pose import catalog
from tests.test_api_jobs import SplitQueue


class ModelStub(PoseEstimator):
    def __init__(self, model):
        self.model = model

    @property
    def landmark_names(self):
        return tuple(f'joint_{i}' for i in range(17 if self.model == 'vitpose' else 33))

    @property
    def landmark_connections(self):
        return ((0, 1),)

    def detect(self, frame_bgr, timestamp_ms=0):
        return (PersonPose(tuple(Landmark(.2 + (i % 2) * .2, .2 + (i % 3) * .1, 0, .9, .9)
                                 for i in range(len(self.landmark_names)))),)


@pytest.fixture
def setup(settings, ingest_video, monkeypatch):
    video = ingest_video(seconds=.3)
    app = create_app()
    queue = SplitQueue(settings)
    choices = []
    monkeypatch.setattr(catalog, 'unavailable_reason', lambda model: None)

    def factory(settings=deps.Depends(deps.get_pose_settings)):
        @contextmanager
        def estimator():
            choices.append(settings.pose_model)
            yield ModelStub(settings.pose_model)
        return estimator

    app.dependency_overrides[deps.get_settings] = lambda: settings
    app.dependency_overrides[deps.get_estimator_factory] = factory
    app.dependency_overrides[deps.get_job_queue] = lambda: queue
    with TestClient(app) as client:
        yield client, video, queue, choices


def pick(client, video, model='heavy', **params):
    response = client.get(f'/videos/{video.id}/candidates', params={'pose_model': model, **params})
    assert response.status_code == 200, response.text
    return response.json()


def submit(client, video, candidate):
    return client.post(f'/videos/{video.id}/analyse', json={
        'candidate_index': 0, 'pose_model': candidate['pose_model'],
        'selection_id': candidate['selection_id'],
    })


def test_config_and_opening_do_not_compute_and_model_cache_is_not_reused(setup, settings):
    client, video, _, choices = setup
    config = client.get('/config').json()
    assert config['default_pose_model'] == settings.pose_model
    assert [m['id'] for m in config['pose_models']] == ['lite', 'full', 'heavy', 'vitpose']
    assert choices == []
    heavy = pick(client, video)
    assert pick(client, video) == heavy
    vitpose = pick(client, video, 'vitpose')
    assert choices == ['heavy', 'vitpose']
    assert vitpose['selection_id'] != heavy['selection_id']
    assert settings.pose_model == 'lite'  # no global mutation
    assert client.get(heavy['frame_url'], params={'selection_id': heavy['selection_id']}).status_code == 409
    assert client.get(vitpose['frame_url'], params={'selection_id': vitpose['selection_id']}).status_code == 200


def test_stale_model_and_same_model_refresh_are_rejected_before_queueing(setup):
    client, video, queue, _ = setup
    first = pick(client, video)
    second = pick(client, video, refresh=True)
    assert submit(client, video, first).status_code == 409
    assert client.post(f'/videos/{video.id}/analyse', json={'candidate_index': 0,
        'pose_model': 'vitpose', 'selection_id': second['selection_id']}).status_code == 409
    assert client.post(f'/videos/{video.id}/analyse', json={'candidate_index': 0,
        'pose_model': 'heavy'}).status_code == 409
    assert queue.enqueued == []


def test_two_models_queue_together_and_worker_uses_each_snapshot(setup, settings, monkeypatch):
    client, video, queue, _ = setup
    heavy = pick(client, video)
    first = submit(client, video, heavy)
    vitpose = pick(client, video, 'vitpose')
    second = submit(client, video, vitpose)
    assert first.status_code == second.status_code == 202
    jobs = [first.json(), second.json()]
    assert queue.enqueued == [j['id'] for j in jobs]
    seen = []

    @contextmanager
    def estimator(**kwargs):
        seen.append(kwargs['variant'])
        yield ModelStub(kwargs['variant'])

    monkeypatch.setattr('app.analysis.create_pose_estimator', estimator)
    # Change candidates again and worker's live default before consuming either job.
    pick(client, video, 'full', frame_index=1)
    handler = AnalysisJobHandler(replace(settings, pose_model='full'))
    for job in jobs:
        handler(job['id'])
    assert seen == ['heavy', 'heavy', 'vitpose', 'vitpose']  # first pass + refinement
    exports = [client.get(f'/videos/{video.id}/keypoints', params={
        'analysis_run_id': j['analysis_run_id']}).json() for j in jobs]
    assert [e['pose_model'] for e in exports] == ['heavy', 'vitpose']
    assert [len(e['landmark_names']) for e in exports] == [33, 17]
    assert all(e['tracked_frame_count'] > 0 for e in exports)
    assert client.get(f'/videos/{video.id}/keypoints').json()['analysis_run_id'] == jobs[1]['analysis_run_id']
    with session_scope() as session:
        repo = SqlAlchemyAnalysisRunRepository(session)
        runs = [repo.get(j['analysis_run_id']) for j in jobs]
        assert runs[0].keypoints_path != runs[1].keypoints_path


def test_unknown_and_uninstalled_models_fail_before_detection_or_queue(setup, monkeypatch):
    client, video, queue, choices = setup
    assert client.get(f'/videos/{video.id}/candidates?pose_model=unknown').status_code == 422
    current = pick(client, video, 'vitpose')
    monkeypatch.setattr(catalog, 'unavailable_reason', lambda name: 'not installed' if name == 'vitpose' else None)
    assert client.get(f'/videos/{video.id}/candidates?pose_model=vitpose').status_code == 503
    assert submit(client, video, current).status_code == 503
    unavailable = client.get('/config').json()['pose_models'][-1]
    assert unavailable['available'] is False
    assert choices == ['vitpose']
    assert queue.enqueued == []


def test_selection_from_another_video_is_rejected(setup, ingest_video):
    client, video, queue, _ = setup
    other = ingest_video(filename='other.mp4', seconds=.3)
    selection = pick(client, video)
    pick(client, other)
    assert submit(client, other, selection).status_code == 409
    assert queue.enqueued == []


def test_legacy_json_and_index_only_default_request_remain_supported(setup, settings):
    client, video, _, _ = setup
    legacy = _decode_candidates(json.dumps({'frame_index': 0, 'candidates': [
        {'index': 0, 'x': .2, 'y': .2, 'width': .2, 'height': .2, 'mean_visibility': .9}]}))
    assert legacy.pose_model is None and legacy.selection_id is None
    with session_scope() as session:
        SqlAlchemyVideoRepository(session).save_candidates(video.id, legacy)
    response = client.post(f'/videos/{video.id}/analyse', json={'candidate_index': 0})
    assert response.status_code == 202
    detail = client.get(f'/videos/{video.id}/analysis-runs/{response.json()["analysis_run_id"]}').json()
    assert detail['pose_configuration']['model'] == settings.pose_model
    detected = pick(client, video, settings.pose_model)
    assert detected['selection_id']  # legacy model-blind cache is not reused


def test_optional_availability_does_not_import_model_runtime(monkeypatch):
    monkeypatch.setattr(catalog, 'find_spec', lambda name: None)
    assert catalog.model_choices()[-1]['available'] is False
    with pytest.raises(catalog.ModelUnavailable):
        catalog.validate_model('vitpose')
    assert catalog.validate_model('heavy') == 'heavy'
