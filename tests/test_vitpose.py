"""Adapter contract tests need neither optional ML dependencies nor weights."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import app.pose as factory
from app.config import Settings
from app.pose.base import RunningMode
from app.pose.vitpose_pose import (
    LANDMARK_CONNECTIONS, LANDMARK_NAMES, VitPoseEstimator, _person_boxes, _to_person,
)


def pose_result(x=100, y=50):
    return {'keypoints': np.tile([x, y], (17, 1)),
            'scores': np.full(17, .7), 'labels': np.arange(17)}


def estimator_stub(boxes=()):
    """Fake only the external model boundary, exercise the real detect method."""
    seen = {}
    estimator = VitPoseEstimator.__new__(VitPoseEstimator)
    estimator._closed, estimator._num_poses = False, 5
    estimator._torch = SimpleNamespace(inference_mode=nullcontext, long=np.int64,
                                      full=lambda shape, value, **_: np.full(shape, value))

    class DetectorProcessor:
        def __call__(self, images, return_tensors):
            seen['rgb'] = images.copy()
            return {'pixels': images}

        def post_process_object_detection(self, output, **kwargs):
            seen['detection_kwargs'] = kwargs
            return [{'boxes': np.array(boxes).reshape(-1, 4),
                     'scores': np.full(len(boxes), .8), 'labels': np.zeros(len(boxes))}]

    class PoseProcessor:
        def __call__(self, images, boxes, return_tensors):
            seen['boxes'] = boxes[0].copy()
            return {'pixels': images}

        def post_process_pose_estimation(self, output, boxes, threshold):
            assert threshold is None  # retain low-confidence joints, stable schema
            return [[pose_result(x + w / 2, y + h / 2) for x, y, w, h in boxes[0]]]

    estimator._detector_processor = DetectorProcessor()
    estimator._detector = lambda **_: None
    estimator._pose_processor = PoseProcessor()

    def predict(**kwargs):
        seen['dataset_index'] = kwargs['dataset_index']

    estimator._pose = predict
    return estimator, seen


def test_multiple_boxes_each_get_a_coco_pose_and_expert_index():
    estimator, seen = estimator_stub(((20, 10, 60, 50), (120, 20, 200, 80)))
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:] = [10, 20, 30]
    people = estimator.detect(frame, 100)
    assert len(people) == 2
    assert [len(p.landmarks) for p in people] == [17, 17]
    assert people[0].landmarks[0].x == pytest.approx(.2)
    assert people[0].landmarks[0].y == pytest.approx(.3)
    assert people[1].landmarks[0].x == pytest.approx(.8)
    assert people[1].landmarks[0].y == pytest.approx(.5)
    assert seen['dataset_index'].tolist() == [0, 0]
    assert seen['boxes'].tolist() == [[20, 10, 40, 40], [120, 20, 80, 60]]
    assert seen['rgb'][0, 0].tolist() == [30, 20, 10]
    assert seen['detection_kwargs']['target_sizes'] == [(100, 200)]
    assert estimator.num_landmarks == 17
    assert all(0 <= i < 17 for edge in LANDMARK_CONNECTIONS for i in edge)


def test_detection_is_stateless_and_empty_boxes_skip_pose():
    estimator, seen = estimator_stub(((20, 10, 60, 50),))
    frame = np.zeros((100, 200, 3), np.uint8)
    assert estimator.detect(frame, 20) == estimator.detect(frame, 20) == estimator.detect(frame, -1)
    empty, seen = estimator_stub()
    assert empty.detect(frame) == ()
    assert 'boxes' not in seen
    empty.close()
    empty.close()
    with pytest.raises(RuntimeError, match='closed'):
        empty.detect(frame)


def test_person_filter_clips_valid_boxes_orders_scores_and_caps_people():
    result = {
        'boxes': [[-5, 0, 80, 120], [10, 20, 30, 60], [1, 1, 5, 5],
                  [2, 2, 7, 7], [30, 30, 10, 10], [0, 0, float('nan'), 20]],
        'scores': [.7, .9, .99, .1, .95, .95],
        'labels': [0, 0, 1, 0, 0, 0],
    }
    assert _person_boxes(result, 100, 100, 5).tolist() == [[10, 20, 20, 40], [0, 0, 80, 100]]
    assert _person_boxes(result, 100, 100, 1).tolist() == [[10, 20, 20, 40]]


def test_full_frame_normalization_reorders_labels_preserves_all_confidences():
    raw = pose_result()
    raw['labels'] = np.arange(16, -1, -1)
    raw['keypoints'][16] = [-10, 700]
    raw['scores'][16] = -.3
    raw['scores'][0] = 1.5
    person = _to_person(raw, 1000, 500)
    assert person.landmarks[0].x == 0
    assert person.landmarks[0].y == 1
    assert person.landmarks[0].visibility == 0
    assert person.landmarks[16].presence == 1
    assert person.landmarks[1].x == .1
    assert person.landmarks[1].y == .1
    assert all(point.z == 0 for point in person.landmarks)


@pytest.mark.parametrize('field,value', [
    ('keypoints', np.zeros((16, 2))), ('labels', np.zeros(17)),
    ('scores', np.full(17, float('nan'))), ('keypoints', np.full((17, 2), float('inf'))),
])
def test_invalid_results_fail_explicitly(field, value):
    raw = pose_result()
    raw[field] = value
    with pytest.raises(ValueError):
        _to_person(raw, 100, 200)


def test_variant_dispatch_never_fetches_mediapipe_task(monkeypatch):
    calls = []
    monkeypatch.setattr('app.pose.vitpose_pose.VitPoseEstimator', lambda **kw: calls.append(kw))
    monkeypatch.setattr(factory, 'ensure_model', lambda *a, **k: pytest.fail('unexpected .task fetch'))
    factory.create_pose_estimator(variant='vitpose', mode=RunningMode.IMAGE, num_poses=2)
    assert calls == [{'mode': RunningMode.IMAGE, 'num_poses': 2}]
    factory.create_pose_estimator(settings=Settings(pose_model='vitpose', max_people=4))
    assert calls[-1]['num_poses'] == 4


def test_existing_default_dispatch_and_missing_optional_dependency(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'app.pose.mediapipe_pose', SimpleNamespace(
        MediaPipePoseEstimator=lambda *a, **kw: calls.append((a, kw))))
    monkeypatch.setattr(factory, 'ensure_model', lambda variant, **kw: variant + '.task')
    factory.create_pose_estimator(settings=Settings(pose_model='lite'))
    assert calls[0][0] == ('lite.task',)
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(ImportError, match='requirements-vitpose'):
        VitPoseEstimator()
    with pytest.raises(ValueError, match='positive'):
        VitPoseEstimator(num_poses=0)
    assert len(LANDMARK_NAMES) == len(set(LANDMARK_NAMES))
