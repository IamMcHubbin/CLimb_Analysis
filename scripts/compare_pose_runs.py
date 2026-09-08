#!/usr/bin/env python3
"""Compare raw API keypoint exports on visually labelled occlusion intervals.

Example (interval ends are exclusive):
    python scripts/compare_pose_runs.py --run heavy=heavy.json \
        --run vitpose=vitpose.json --segments segments.json \
        --width 720 --height 1280 --json comparison.json

segments.json is a list, for example:
    [{"label": "left arm behind torso", "start_frame": 300, "end_frame": 345,
      "occluded_joints": ["left_elbow", "left_wrist"]}]

Inputs are unmodified GET /videos/{id}/keypoints?analysis_run_id=... responses
or outputs of replay_pose_detections.py, from the same normalised clip.
Do not mix native results and shared-schema replays in one comparison.
Width and height are that clip's dimensions,
not the source phone file's. No confidence threshold or smoothing is applied.
Coverage, box consistency and jitter are proxies, not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


SHARED_JOINTS = tuple(
    f"{side}_{joint}"
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    for side in ("left", "right")
)


@dataclass(frozen=True)
class Segment:
    label: str
    start_frame: int
    end_frame: int
    occluded_joints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label or self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("segments need a label and 0 <= start_frame < end_frame")
        if len(set(self.occluded_joints)) != len(self.occluded_joints):
            raise ValueError("occluded_joints must be unique")
        if set(self.occluded_joints) - set(SHARED_JOINTS):
            raise ValueError("occluded_joints must belong to the shared 12-joint schema")


def _distribution(values) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "median": None, "p90": None, "p99": None}

    def percentile(fraction):
        # Linear interpolation, explicitly fixed so different NumPy versions
        # or the timing script's nearest-rank method cannot alter this report.
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"samples": len(ordered), "median": statistics.median(ordered),
            "p90": percentile(0.90), "p99": percentile(0.99)}


def _shared_frames(payload: dict, width: int, height: int) -> list:
    names = payload["landmark_names"]
    if len(names) != len(set(names)) or set(SHARED_JOINTS) - set(names):
        raise ValueError("each run needs unique landmark names and all 12 shared joints")
    if len(payload["frames"]) != payload["frame_count"]:
        raise ValueError("frames must be index-aligned with frame_count")
    if len(payload["match_iou"]) != payload["frame_count"]:
        raise ValueError("match_iou must be index-aligned with frame_count")
    indices = [names.index(name) for name in SHARED_JOINTS]
    result = []
    for frame in payload["frames"]:
        if frame is None:
            result.append(None)
            continue
        if len(frame) != len(names):
            raise ValueError("a tracked frame's landmark count does not match its schema")
        # Leave extrapolated coordinates intact for jitter. Confidence scales
        # differ between estimators and do not identify actual visibility.
        result.append(tuple((float(frame[i][0]) * width, float(frame[i][1]) * height)
                            for i in indices))
    return result


def _is_finite(frame) -> bool:
    return frame is not None and all(math.isfinite(v) for point in frame for v in point)


def _box(frame, width: int, height: int):
    # Same clamping convention as PersonPose.bounding_box, using only shared
    # joints. No replay of the tracker: this is adjacent-frame box consistency.
    xs = [min(width, max(0.0, point[0])) for point in frame]
    ys = [min(height, max(0.0, point[1])) for point in frame]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return min(1.0, intersection / union) if union > 0 else 0.0


def _segment_samples(payload: dict, frames: list, segment: Segment,
                     width: int, height: int) -> dict:
    tracked = {i for i in range(segment.start_frame, segment.end_frame)
               if frames[i] is not None}
    native = {i: float(payload["match_iou"][i]) for i in tracked
              if payload["match_iou"][i] is not None
              and math.isfinite(float(payload["match_iou"][i]))}
    heights = {i: max(p[1] for p in frames[i]) - min(p[1] for p in frames[i])
               for i in tracked if _is_finite(frames[i])}
    adjacent = {}
    jitter = {}
    for i in range(segment.start_frame, segment.end_frame):
        if i > segment.start_frame and _is_finite(frames[i - 1]) and _is_finite(frames[i]):
            adjacent[i] = _iou(_box(frames[i - 1], width, height),
                               _box(frames[i], width, height))
        if i == segment.start_frame or i + 1 == segment.end_frame:
            continue
        if (heights.get(i, 0.0) <= 0 or not _is_finite(frames[i - 1])
                or not _is_finite(frames[i + 1])):
            continue
        jitter[i] = tuple(
            math.hypot(after[0] - 2 * centre[0] + before[0],
                       after[1] - 2 * centre[1] + before[1]) / heights[i]
            for before, centre, after in zip(frames[i - 1], frames[i], frames[i + 1])
        )
    return {"tracked": tracked, "native": native, "heights": heights,
            "adjacent": adjacent, "jitter": jitter}


def _jitter_report(samples: dict, segment: Segment, centers: set | None = None) -> dict:
    selected = {i: values for i, values in samples["jitter"].items()
                if centers is None or i in centers}
    report = {
        "center_frame_indices": sorted(selected),
        "valid_centers": len(selected),
        "possible_centers": max(0, segment.end_frame - segment.start_frame - 2),
        "shared_joints": _distribution(statistics.median(values) for values in selected.values()),
        "per_joint": {name: _distribution(values[j] for values in selected.values())
                      for j, name in enumerate(SHARED_JOINTS)},
    }
    report["skipped_centers"] = report["possible_centers"] - len(selected)
    if segment.occluded_joints:
        indices = [SHARED_JOINTS.index(name) for name in segment.occluded_joints]
        report["occluded_joints"] = _distribution(
            statistics.median(values[j] for j in indices) for values in selected.values())
    return report


def compare_runs(runs: dict[str, dict], segments: list[Segment], *,
                 width: int, height: int, paired: bool = True) -> dict:
    """Return JSON-serialisable diagnostics; never compare missing frames as zeros."""
    if not runs or width <= 0 or height <= 0:
        raise ValueError("provide at least one run and positive normalised video dimensions")
    if not segments or len({segment.label for segment in segments}) != len(segments):
        raise ValueError("provide segments with distinct labels")
    first = next(iter(runs.values()))
    replayed = bool(first.get('diagnostic'))
    for payload in runs.values():
        if bool(payload.get('diagnostic')) != replayed:
            raise ValueError('do not mix native results and diagnostic replays')
        if any(payload[key] != first[key] for key in ("video_id", "fps", "frame_count")):
            raise ValueError("runs must belong to the same video, frame count and fps")
    if any(segment.end_frame > first["frame_count"] for segment in segments):
        raise ValueError("segment exceeds the normalised clip's frame count")
    frames = {name: _shared_frames(payload, width, height) for name, payload in runs.items()}
    result = {
        "video_id": first["video_id"], "fps": first["fps"], "width": width, "height": height,
        "shared_joints": SHARED_JOINTS,
        "method": {
            "intervals": "start inclusive, end exclusive; never bridge gaps or interval boundaries",
            "tracked_percent": "100 * non-null frames / interval frames; no visibility threshold",
            "native_match_iou": (
                "unchanged tracker score from shared-12-joint replay, before refinement"
                if replayed else
                "stored tracker score, based on each model's full schema before refinement; not schema-equivalent"),
            "shared_box_iou": "consecutive available frames' clamped 12-joint bounding-box IoU; not a replayed tracker score",
            "jitter": "median across selected joints of ||p[t+1] - 2*p[t] + p[t-1]||_2 / shared_body_height[t]; p in pixels",
            "body_height": "max(y)-min(y) over all 12 shared joints in the centre frame, in pixels, without confidence filtering",
            "percentiles": "linear interpolation over frame medians; no smoothing, fps-squared scaling or gap interpolation",
            "paired": "same valid three-frame centers (and adjacent pairs) in every compared run; read alongside unpaired coverage",
            "limitation": "coverage and temporal consistency are not joint accuracy; a confidently wrong or static pose can score well",
            "precision": "API exports round x/y to 4 decimals and stored match IoU to 3 decimals",
        },
        "runs": {name: {"analysis_run_id": payload.get("analysis_run_id"),
                        "pose_model": payload.get("pose_model")}
                 for name, payload in runs.items()},
        "segments": [],
    }
    for segment in segments:
        samples = {name: _segment_samples(payload, frames[name], segment, width, height)
                   for name, payload in runs.items()}
        paired_centers = set.intersection(*(set(sample["jitter"]) for sample in samples.values()))
        paired_adjacent = set.intersection(*(set(sample["adjacent"]) for sample in samples.values()))
        entry = {"label": segment.label, "start_frame": segment.start_frame,
                 "end_frame": segment.end_frame, "occluded_joints": segment.occluded_joints,
                 "runs": {}}
        for name, sample in samples.items():
            count = segment.end_frame - segment.start_frame
            metrics = {
                "frames": count, "tracked_frames": len(sample["tracked"]),
                "tracked_percent": 100 * len(sample["tracked"]) / count,
                "native_match_iou": _distribution(sample["native"].values()),
                "shared_box_iou": _distribution(sample["adjacent"].values()),
                "shared_body_height_pixels": _distribution(sample["heights"].values()),
                "jitter": _jitter_report(sample, segment),
            }
            if paired:
                metrics["paired_jitter"] = _jitter_report(sample, segment, paired_centers)
                metrics["paired_shared_box_iou"] = _distribution(
                    sample["adjacent"][i] for i in paired_adjacent)
            entry["runs"][name] = metrics
        result["segments"].append(entry)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSON_PATH")
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True, help="normalised video width")
    parser.add_argument("--height", type=int, required=True, help="normalised video height")
    parser.add_argument("--no-paired", action="store_true", help="omit diagnostics on paired available frames")
    parser.add_argument("--json", type=Path, help="write the full report as JSON as well as stdout")
    args = parser.parse_args(argv)
    runs = {}
    for argument in args.run:
        name, separator, filename = argument.partition("=")
        if not separator or not name or name in runs:
            parser.error("each --run must have a distinct LABEL=JSON_PATH")
        runs[name] = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    segments = [Segment(**item) for item in json.loads(args.segments.read_text(encoding="utf-8-sig"))]
    result = compare_runs(runs, segments, width=args.width, height=args.height, paired=not args.no_paired)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
