#!/usr/bin/env python3
"""Run the unchanged app locally, recording estimator inputs/outputs for an experiment.

Use a separate --data-dir; never point this at the ordinary application's data.
Records include all people, not just the chosen track, so a shared-schema
replay can use the existing tracker without changing application code.
No public tunnel is started. Ctrl+C stops this experimental server.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('lite', 'full', 'heavy', 'vitpose'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--records-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--torch-threads', type=int, default=4)
    args = parser.parse_args()
    os.environ['CLIMB_POSE_MODEL'] = args.model
    os.environ['CLIMB_DATA_DIR'] = str(args.data_dir.resolve())
    os.environ['CLIMB_RETAIN_ANALYSED_SECONDS'] = '604800'
    os.environ['CLIMB_RETAIN_UNANALYSED_SECONDS'] = '604800'
    args.records_dir.mkdir(parents=True, exist_ok=True)
    if args.model == 'vitpose':
        import torch
        torch.set_num_threads(args.torch_threads)

    import app.pose as pose
    from app.pose.base import PoseEstimator
    original_factory = pose.create_pose_estimator

    class RecordedEstimator(PoseEstimator):
        def __init__(self, inner, handle):
            self.inner, self.handle = inner, handle
            self.calls = 0

        @property
        def landmark_names(self):
            return self.inner.landmark_names

        @property
        def landmark_connections(self):
            return self.inner.landmark_connections

        def detect(self, frame_bgr, timestamp_ms=0):
            started = time.perf_counter()
            people = self.inner.detect(frame_bgr, timestamp_ms)
            elapsed = time.perf_counter() - started
            self.handle.write(json.dumps({
                'call': self.calls, 'timestamp_ms': timestamp_ms,
                'width': frame_bgr.shape[1], 'height': frame_bgr.shape[0],
                'inference_seconds': elapsed,
                'people': [[[p.x, p.y, p.z, p.visibility, p.presence]
                            for p in person.landmarks] for person in people],
            }, allow_nan=False) + '\n')
            self.calls += 1
            return people

        def close(self):
            try:
                self.inner.close()
            finally:
                self.handle.close()

    def recorded_factory(**kwargs):
        started = time.perf_counter()
        inner = original_factory(**kwargs)
        model = kwargs.get('variant') or getattr(kwargs.get('settings'), 'pose_model', args.model)
        # Unique time-based path; never overwrite an earlier recording.
        destination = args.records_dir / f'{model}-{time.time_ns()}.jsonl.gz'
        handle = gzip.open(destination, 'wt', encoding='utf-8')
        handle.write(json.dumps({
            'model': model, 'mode': kwargs.get('mode', pose.RunningMode.VIDEO).value,
            'landmark_names': inner.landmark_names,
            'landmark_connections': inner.landmark_connections,
            'load_seconds': time.perf_counter() - started,
        }) + '\n')
        print(f'Estimator recording: {destination}', flush=True)
        return RecordedEstimator(inner, handle)

    # Applied before app.main imports its factory bindings. The actual app,
    # candidates, job handler, tracker, refinement and storage run unchanged.
    pose.create_pose_estimator = recorded_factory
    import uvicorn
    uvicorn.run('app.main:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
