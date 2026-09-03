#!/usr/bin/env python3
"""Time pose estimation over a clip.

Answers the question the rest of the build depends on: how long does analysing
one climbing video actually take on CPU, and how much does the model variant
change that.

Decode and inference are timed separately because they scale differently -
decode with resolution, inference with model size and the number of people
being tracked.

Usage:
    python scripts/benchmark_pose.py --video clip.mp4
    python scripts/benchmark_pose.py --video clip.mp4 --models lite,full,heavy \
        --num-poses 5 --normalise --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow running as `python scripts/benchmark_pose.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.ingest.normalise import normalise_video  # noqa: E402
from app.ingest.probe import probe_video  # noqa: E402
from app.pose import RunningMode, create_pose_estimator  # noqa: E402


@dataclass
class BenchmarkResult:
    model: str
    num_poses: int
    frames: int
    width: int
    height: int
    clip_fps: float
    clip_seconds: float

    decode_total_s: float
    inference_total_s: float
    wall_total_s: float

    inference_ms_mean: float
    inference_ms_p50: float
    inference_ms_p90: float
    inference_ms_p99: float
    inference_ms_max: float

    processed_fps: float
    realtime_factor: float
    frames_with_detection: int
    detection_rate: float
    mean_people_per_frame: float

    # Repeats guard against a noisy machine. The reported run is the fastest
    # one; ``repeat_p50_ms`` shows every repeat so the noise floor is visible.
    repeats: int = 1
    repeat_p50_ms: tuple[float, ...] = ()

    def summary_lines(self) -> list[str]:
        return [
            f"model                {self.model} (num_poses={self.num_poses})",
            f"clip                 {self.width}x{self.height}, {self.frames} frames, "
            f"{self.clip_seconds:.1f}s @ {self.clip_fps:g}fps",
            f"wall time            {self.wall_total_s:.1f}s "
            f"(decode {self.decode_total_s:.1f}s, inference {self.inference_total_s:.1f}s)",
            f"inference per frame  mean {self.inference_ms_mean:.1f}ms | "
            f"p50 {self.inference_ms_p50:.1f} | p90 {self.inference_ms_p90:.1f} | "
            f"p99 {self.inference_ms_p99:.1f} | max {self.inference_ms_max:.1f}",
            f"throughput           {self.processed_fps:.1f} fps "
            f"({self.realtime_factor:.2f}x realtime)",
            f"detections           {self.detection_rate * 100:.1f}% of frames, "
            f"{self.mean_people_per_frame:.2f} people/frame avg",
            f"repeats              {self.repeats} run(s), per-run p50 "
            + " / ".join(f"{value:.1f}" for value in self.repeat_p50_ms)
            + "ms (fastest reported)",
            f"projection           a 60s clip takes ~{60.0 / max(self.realtime_factor, 1e-9):.0f}s "
            f"to analyse",
        ]


def _prefetch(path: Path) -> None:
    """Pull the file into the page cache before timing.

    Without this the first model measured pays the disk read for the whole
    clip and looks slower than the ones after it - which is how this script
    originally reported lite as slower than full.
    """
    with path.open("rb") as handle:
        while handle.read(4 << 20):
            pass


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def benchmark(
    video_path: Path,
    model: str,
    *,
    num_poses: int,
    max_frames: int | None,
    mode: RunningMode,
    warmup_frames: int,
) -> BenchmarkResult:
    probe = probe_video(video_path)
    _prefetch(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open {video_path}")

    estimator = create_pose_estimator(mode=mode, variant=model, num_poses=num_poses)

    decode_seconds = 0.0
    inference_times_ms: list[float] = []
    frames = 0
    frames_with_detection = 0
    total_people = 0
    fps = probe.fps or 30.0

    try:
        # The first calls allocate buffers and warm the delegate; including
        # them would misrepresent steady-state cost, so they are timed and
        # discarded. Timestamps keep climbing across the reset: in VIDEO mode
        # the estimator rejects a timestamp that does not advance, and clamping
        # it would change the tracking behaviour being measured.
        timestamp_frame = 0
        for _ in range(warmup_frames):
            ok, frame = capture.read()
            if not ok:
                break
            estimator.detect(frame, int(round(1000 * timestamp_frame / fps)))
            timestamp_frame += 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        wall_start = time.perf_counter()
        while max_frames is None or frames < max_frames:
            decode_start = time.perf_counter()
            ok, frame = capture.read()
            decode_seconds += time.perf_counter() - decode_start
            if not ok:
                break

            timestamp_ms = int(round(1000 * timestamp_frame / fps))
            inference_start = time.perf_counter()
            people = estimator.detect(frame, timestamp_ms)
            inference_times_ms.append((time.perf_counter() - inference_start) * 1000)

            timestamp_frame += 1
            frames += 1
            if people:
                frames_with_detection += 1
                total_people += len(people)
        wall_seconds = time.perf_counter() - wall_start
    finally:
        capture.release()
        estimator.close()

    if frames == 0:
        raise SystemExit(f"no frames decoded from {video_path}")

    inference_total = sum(inference_times_ms) / 1000
    processed_fps = frames / wall_seconds if wall_seconds > 0 else 0.0

    return BenchmarkResult(
        model=model,
        num_poses=num_poses,
        frames=frames,
        width=probe.display_width,
        height=probe.display_height,
        clip_fps=fps,
        clip_seconds=frames / fps,
        decode_total_s=decode_seconds,
        inference_total_s=inference_total,
        wall_total_s=wall_seconds,
        inference_ms_mean=statistics.fmean(inference_times_ms),
        inference_ms_p50=_percentile(inference_times_ms, 0.50),
        inference_ms_p90=_percentile(inference_times_ms, 0.90),
        inference_ms_p99=_percentile(inference_times_ms, 0.99),
        inference_ms_max=max(inference_times_ms),
        processed_fps=processed_fps,
        # >1 means analysis keeps up with playback; <1 means it falls behind.
        realtime_factor=processed_fps / fps if fps else 0.0,
        frames_with_detection=frames_with_detection,
        detection_rate=frames_with_detection / frames,
        mean_people_per_frame=total_people / frames,
    )


def benchmark_repeated(
    video_path: Path,
    model: str,
    *,
    repeats: int,
    **kwargs,
) -> BenchmarkResult:
    """Run the benchmark several times and report the fastest run.

    Shared machines produce runs two to five times slower than the hardware is
    capable of, at random. Averaging bakes that in; taking the fastest run
    estimates the real cost, and the per-run spread reported alongside it says
    how much to trust the number.
    """
    runs = [benchmark(video_path, model, **kwargs) for _ in range(max(1, repeats))]
    best = min(runs, key=lambda result: result.inference_ms_p50)
    best.repeats = len(runs)
    best.repeat_p50_ms = tuple(result.inference_ms_p50 for result in runs)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", default="lite", help="comma separated: lite,full,heavy")
    parser.add_argument("--num-poses", type=int, default=settings.max_people)
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--repeats", type=int, default=3,
                        help="measurement passes per model; the fastest is reported")
    parser.add_argument("--warmup-frames", type=int, default=30,
                        help="frames to run before timing starts, to reach steady state")
    parser.add_argument("--mode", choices=[m.value for m in RunningMode], default=RunningMode.VIDEO.value)
    parser.add_argument("--normalise", action="store_true",
                        help="run ingest normalisation first and benchmark the result")
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = parser.parse_args(argv)

    if not args.video.exists():
        raise SystemExit(f"no such file: {args.video}")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = args.video
        normalise_seconds = None
        if args.normalise:
            destination = Path(tmp) / "normalised.mp4"
            started = time.perf_counter()
            normalised = normalise_video(args.video, destination)
            normalise_seconds = time.perf_counter() - started
            video_path = normalised.path
            print(
                f"normalised in {normalise_seconds:.1f}s -> "
                f"{normalised.width}x{normalised.height}, {normalised.frame_count} frames "
                f"@ {normalised.fps:g}fps\n"
            )

        results = []
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            result = benchmark_repeated(
                video_path,
                model,
                repeats=args.repeats,
                num_poses=args.num_poses,
                max_frames=args.max_frames,
                mode=RunningMode(args.mode),
                warmup_frames=args.warmup_frames,
            )
            results.append(result)
            print("\n".join(result.summary_lines()))
            print()

    if args.json:
        payload = {
            "video": str(args.video),
            "normalise_seconds": normalise_seconds,
            "results": [asdict(result) for result in results],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
