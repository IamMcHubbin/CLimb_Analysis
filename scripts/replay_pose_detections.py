#!/usr/bin/env python3
"""Replay recorded detections through the existing IoU tracker on 12 shared joints.

Input is a first (full-frame) VIDEO pass recorded by serve_pose_experiment.py.
No inference or interpolation occurs. The same supplied seed box and tracking
configuration must be used for every model. Outputs match the keypoint API
shape so compare_pose_runs.py can quantify schema-equivalent association.
This is a diagnostic replay, not a replacement for the native application run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.geometry import BoundingBox
from app.pose.base import Landmark, PersonPose
from app.tracking import IouTracker, TrackingConfig
from scripts.compare_pose_runs import SHARED_JOINTS


def replay(header, records, *, seed_frame, seed_box, frame_count, fps, video_id,
           config=None):
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('frame count and fps must be positive')
    if header['mode'] != 'video' or len(records) != frame_count:
        raise ValueError('recording must contain one full VIDEO pass, not a picker/refinement pass')
    if not 0 <= seed_frame < frame_count:
        raise ValueError('seed frame outside clip')
    indices = [header['landmark_names'].index(name) for name in SHARED_JOINTS]
    dimensions = {(row['width'], row['height']) for row in records}
    if len(dimensions) != 1:
        raise ValueError('recording has crop dimensions, expected full-frame detections')
    detections = []
    for index, row in enumerate(records):
        if row['call'] != index or abs(row['timestamp_ms'] - round(index * 1000 / fps)) > 1:
            raise ValueError('recording must be consecutive original video frames')
        detections.append(tuple(PersonPose(tuple(Landmark(*person[i]) for i in indices))
                                for person in row['people']))
    config = config or TrackingConfig()
    results = {}
    for sequence in (range(seed_frame, frame_count), range(seed_frame - 1, -1, -1)):
        tracker = IouTracker(seed_box, config)
        for index in sequence:
            results[index] = tracker.update(index, detections[index])
    return {
        'video_id': video_id, 'fps': fps, 'frame_count': frame_count,
        'pose_model': header['model'], 'analysis_run_id': None,
        'landmark_names': list(SHARED_JOINTS), 'landmark_connections': [],
        'seed_frame': seed_frame, 'seed_box': vars(seed_box),
        'tracking_config': vars(config), 'diagnostic': 'shared-joint replay, no refinement',
        'frames': [None if results[i].is_gap else
                   [[round(p.x, 4), round(p.y, 4), round(p.z, 4),
                     round(p.visibility, 4), round(p.presence, 4)]
                    for p in results[i].person.landmarks] for i in range(frame_count)],
        'match_iou': [None if results[i].is_gap else round(results[i].iou, 3)
                      for i in range(frame_count)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--seed-frame', type=int, required=True)
    parser.add_argument('--seed-box', type=float, nargs=4, required=True, metavar=('X','Y','W','H'))
    parser.add_argument('--frame-count', type=int, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.recording, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    result = replay(header, rows, seed_frame=args.seed_frame, seed_box=BoundingBox(*args.seed_box),
                    frame_count=args.frame_count, fps=args.fps, video_id=args.video_id)
    args.json.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(f'Wrote {args.json}: {sum(f is not None for f in result["frames"])}/{args.frame_count} tracked')


if __name__ == '__main__':
    main()
