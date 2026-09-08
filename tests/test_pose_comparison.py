from __future__ import annotations

import copy
import json

import pytest

from scripts.compare_pose_runs import SHARED_JOINTS, Segment, compare_runs, main


def _run(offsets=(0.0, 0.01, 0.04), *, extra_names=(), missing=()):
    names = list(SHARED_JOINTS) + list(extra_names)
    frames = []
    for index, offset in enumerate(offsets):
        # Fixed 0.5-frame-height extent, with acceleration only in x. At
        # 1000x500 the [0,.01,.04] translation is .02*1000/250 == .08 jitter.
        frame = [[0.2 + 0.04 * (joint % 2) + offset,
                  0.2 + 0.1 * (joint // 2), 0.0]
                 for joint in range(len(SHARED_JOINTS))]
        frame.extend([[100 * (index + 1), -100 * (index + 1), 1.0] for _ in extra_names])
        frames.append(None if index in missing else frame)
    return {"video_id": "video", "fps": 30, "frame_count": len(frames),
            "analysis_run_id": "run", "pose_model": "test", "landmark_names": names,
            "frames": frames, "match_iou": [None if i in missing else 0.8 for i in range(len(frames))]}


def _compare(run, segments=None, **kwargs):
    return compare_runs({"test": run}, segments or [Segment("occlusion", 0, run["frame_count"])],
                        width=1000, height=500, **kwargs)["segments"][0]["runs"]["test"]


def test_jitter_converts_axes_to_pixels_and_ignores_nonshared_joints_and_confidence():
    metrics = _compare(_run(extra_names=("left_index", "right_index", "nose")))
    assert metrics["tracked_percent"] == 100
    assert metrics["jitter"]["shared_joints"]["median"] == pytest.approx(0.08)
    assert metrics["jitter"]["shared_joints"]["samples"] == 1
    assert metrics["native_match_iou"]["median"] == 0.8
    assert metrics["shared_body_height_pixels"]["median"] == pytest.approx(250)


def test_joint_names_map_independently_of_schema_order():
    original = _run()
    reversed_schema = copy.deepcopy(original)
    reversed_schema["landmark_names"].reverse()
    for frame in reversed_schema["frames"]:
        frame.reverse()
    assert _compare(original) == _compare(reversed_schema)


def test_gaps_and_interval_boundaries_are_never_bridged():
    run = _run(offsets=(0, .01, .04, .09, .16, .25, .36), missing=(2,))
    metrics = _compare(run, [Segment("occlusion", 1, 6)])
    assert metrics["tracked_frames"] == 4
    assert metrics["tracked_percent"] == 80
    assert metrics["native_match_iou"]["samples"] == 4
    assert metrics["shared_box_iou"]["samples"] == 2  # pairs 3->4 and 4->5
    assert metrics["jitter"]["center_frame_indices"] == [4]
    assert metrics["jitter"]["possible_centers"] == 3
    assert metrics["jitter"]["skipped_centers"] == 2


def test_adjacent_intervals_do_not_create_extra_jitter_centers():
    run = _run(offsets=(0, .01, .04, .09, .16, .25))
    output = compare_runs({"test": run}, [Segment("first", 0, 3), Segment("second", 3, 6)],
                          width=1000, height=500)
    assert [entry["runs"]["test"]["jitter"]["center_frame_indices"]
            for entry in output["segments"]] == [[1], [4]]


def test_empty_and_degenerate_samples_return_null_not_perfect_zero():
    gaps = _compare(_run(missing=(0, 1, 2)))
    assert gaps["tracked_percent"] == 0
    assert gaps["jitter"]["shared_joints"]["median"] is None
    assert gaps["native_match_iou"]["median"] is None
    run = _run()
    for frame in run["frames"]:
        for point in frame:
            point[1] = .5
    assert _compare(run)["jitter"]["valid_centers"] == 0


def test_nonfinite_coordinates_are_skipped_without_hiding_coverage():
    run = _run()
    run["frames"][1][0][0] = float("nan")
    metrics = _compare(run)
    assert metrics["tracked_percent"] == 100
    assert metrics["jitter"]["valid_centers"] == 0
    assert metrics["shared_box_iou"]["samples"] == 0
    json.dumps(metrics, allow_nan=False)


def test_occluded_joint_subset_exposes_errors_hidden_by_whole_body_median():
    run = _run(offsets=(0, 0, 0))
    elbow = SHARED_JOINTS.index("left_elbow")
    run["frames"][1][elbow][0] += .1
    metrics = _compare(run, [Segment("left arm", 0, 3, ("left_elbow",))])
    assert metrics["jitter"]["shared_joints"]["median"] == 0
    assert metrics["jitter"]["occluded_joints"]["median"] == pytest.approx(.8)
    assert metrics["jitter"]["per_joint"]["left_elbow"]["median"] == pytest.approx(.8)


def test_paired_jitter_uses_same_valid_centers_without_erasing_unpaired_coverage():
    runs = {"complete": _run(offsets=(0, .01, .04, .09, .16)),
            "partial": _run(offsets=(0, .01, .04, .09, .16), missing=(0,))}
    metrics = compare_runs(runs, [Segment("turn", 0, 5)], width=1000, height=500)["segments"][0]["runs"]
    assert metrics["complete"]["tracked_percent"] == 100
    assert metrics["partial"]["tracked_percent"] == 80
    assert metrics["complete"]["jitter"]["valid_centers"] == 3
    for run in metrics.values():
        assert run["paired_jitter"]["center_frame_indices"] == [2, 3]
        assert run["paired_shared_box_iou"]["samples"] == 3


@pytest.mark.parametrize("key,value", [("video_id", "different"), ("fps", 24), ("frame_count", 4)])
def test_incompatible_run_provenance_is_rejected(key, value):
    a, b = _run(), _run()
    b[key] = value
    with pytest.raises(ValueError, match="same video"):
        compare_runs({"a": a, "b": b}, [Segment("turn", 0, 3)], width=1000, height=500)


def test_bad_interval_schema_and_frame_alignment_are_rejected():
    with pytest.raises(ValueError, match="shared 12"):
        Segment("bad", 0, 3, ("nose",))
    with pytest.raises(ValueError, match="exceeds"):
        _compare(_run(), [Segment("bad", 0, 4)])
    run = _run()
    run["frames"].pop()
    with pytest.raises(ValueError, match="index-aligned"):
        _compare(run)
    run = _run()
    run["landmark_names"][0] = "nose"
    with pytest.raises(ValueError, match="12 shared"):
        _compare(run)


def test_cli_loads_api_exports_and_writes_report(tmp_path, capsys):
    run_path = tmp_path / "run.json"
    segment_path = tmp_path / "segments.json"
    output_path = tmp_path / "metrics.json"
    run_path.write_text(json.dumps(_run()), encoding="utf-8")
    segment_path.write_text(json.dumps([{"label": "turn", "start_frame": 0, "end_frame": 3}]),
                            encoding="utf-8")
    assert main(["--run", f"heavy={run_path}", "--segments", str(segment_path),
                 "--width", "1000", "--height", "500", "--json", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report == json.loads(capsys.readouterr().out)
    assert report["segments"][0]["runs"]["heavy"]["tracked_percent"] == 100


def test_benchmark_start_window_preserves_video_timestamp_order(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import benchmark_pose

    positions = []
    detected = []
    closed = []

    class Capture:
        def isOpened(self):
            return True

        def set(self, prop, value):
            self.position = value
            positions.append(value)

        def read(self):
            frame = self.position
            self.position += 1
            return True, frame

        def release(self):
            closed.append("capture")

    class Estimator:
        def detect(self, frame, timestamp):
            detected.append((frame, timestamp))
            return ()

        def close(self):
            closed.append("estimator")

    monkeypatch.setattr(benchmark_pose, "probe_video", lambda _: SimpleNamespace(
        fps=30, display_width=1000, display_height=500))
    monkeypatch.setattr(benchmark_pose, "_prefetch", lambda _: None)
    monkeypatch.setattr(benchmark_pose.cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(benchmark_pose, "create_pose_estimator", lambda **_: Estimator())
    result = benchmark_pose.benchmark(tmp_path / "clip.mp4", "vitpose", num_poses=5,
                                     max_frames=3, mode=benchmark_pose.RunningMode.VIDEO,
                                     warmup_frames=2, start_frame=50)
    assert positions == [50, 50]
    assert [frame for frame, _ in detected] == [50, 51, 50, 51, 52]
    assert all(after[1] > before[1] for before, after in zip(detected, detected[1:]))
    assert result.start_frame == 50
    assert result.frames == 3
    assert closed == ["capture", "estimator"]
