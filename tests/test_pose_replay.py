"""A schema-equivalent diagnostic must still use the real greedy tracker."""
import copy

import pytest

from app.geometry import BoundingBox
from app.tracking import TrackingConfig
from scripts.compare_pose_runs import SHARED_JOINTS
from scripts.replay_pose_detections import replay


def recording():
    names = list(reversed(SHARED_JOINTS)) + ['nose']
    # The extra landmark is deliberately far away: it must not affect matching.
    person = [[.4 + .2 * (i % 2), .3 + .4 * (i // 6), 0, .8, .8]
              for i in range(12)] + [[0, 0, 0, 1, 1]]
    rows = [{'call': i, 'timestamp_ms': round(i * 1000 / 30),
             'width': 720, 'height': 1280, 'people': [copy.deepcopy(person)]}
            for i in range(5)]
    return {'mode': 'video', 'model': 'fake', 'landmark_names': names}, rows


def run(header, rows, **kwargs):
    return replay(header, rows, seed_frame=2, seed_box=BoundingBox(.4, .3, .2, .4),
                  frame_count=5, fps=30, video_id='clip', **kwargs)


def test_replay_uses_shared_schema_and_both_directions_without_mutating_input():
    header, rows = recording()
    original = copy.deepcopy(rows)
    result = run(header, rows)
    assert all(frame is not None for frame in result['frames'])
    assert result['match_iou'] == [1] * 5
    assert result['landmark_names'] == list(SHARED_JOINTS)
    assert result['frames'][0][0] == [round(v, 4) for v in rows[0]['people'][0][11]]
    assert len(result['frames'][0]) == 12
    assert rows == original


def test_replay_retains_gaps_reacquires_and_matches_person_not_array_index():
    header, rows = recording()
    other = [[.05, .05, 0, 1, 1]] * 13
    rows[0]['people'].insert(0, other)
    rows[3]['people'] = [other]
    result = run(header, rows)
    assert result['frames'][0] is not None
    assert result['frames'][3] is None
    assert result['match_iou'][3] is None
    assert result['frames'][4] is not None
    limited = run(header, rows, config=TrackingConfig(max_gap_frames=0))
    assert limited['frames'][4] is None


@pytest.mark.parametrize('invalid', ['image', 'length', 'crop', 'timestamp', 'order'])
def test_picker_crop_and_nonconsecutive_recordings_are_rejected(invalid):
    header, rows = recording()
    if invalid == 'image':
        header['mode'] = 'image'
    elif invalid == 'length':
        rows.pop()
    elif invalid == 'crop':
        rows[0]['width'] = 100
    elif invalid == 'timestamp':
        rows[0]['timestamp_ms'] = 99
    else:
        rows[0]['call'] = 2
    with pytest.raises(ValueError):
        run(header, rows)
