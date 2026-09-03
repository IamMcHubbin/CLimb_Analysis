#!/usr/bin/env python3
"""Build a synthetic test clip from a still photo of a person.

Real climbing footage is the point of this project, but a reproducible clip is
needed to exercise the ingest path and to give the benchmark something to chew
on. This pans and zooms across a photo so a person is present and moving in
every frame, and can deliberately reproduce the two things phone video does
that break naive pipelines:

    --rotate 90    write a display matrix, as a phone held sideways does
    --vfr          jitter the frame timestamps, as phone capture does

Usage:
    python scripts/make_sample_clip.py --image person.jpg --out clip.mp4 \
        --seconds 8 --rotate 90 --vfr
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _ken_burns_frame(image: np.ndarray, out_w: int, out_h: int, phase: float) -> np.ndarray:
    """Crop a moving, zooming window out of ``image`` and fit it to the output.

    ``phase`` runs 0..1 across the clip. The motion is two sine waves at
    different rates so the path does not simply retrace itself.
    """
    src_h, src_w = image.shape[:2]
    zoom = 0.62 + 0.14 * math.sin(2 * math.pi * phase)
    crop_w = min(src_w, int(src_w * zoom))
    crop_h = min(src_h, int(crop_w * out_h / out_w))
    if crop_h > src_h:
        crop_h = src_h
        crop_w = min(src_w, int(crop_h * out_w / out_h))

    max_x = max(0, src_w - crop_w)
    max_y = max(0, src_h - crop_h)
    x = int(max_x * (0.5 + 0.5 * math.sin(2 * math.pi * phase * 1.3)))
    y = int(max_y * (0.5 + 0.5 * math.sin(2 * math.pi * phase * 0.7 + 1.0)))

    crop = image[y : y + crop_h, x : x + crop_w]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def render_clip(image_path: Path, destination: Path, width: int, height: int, fps: int, seconds: float) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read image: {image_path}")

    frame_count = max(1, int(round(fps * seconds)))
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(destination),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for index in range(frame_count):
            frame = _ken_burns_frame(image, width, height, index / frame_count)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed while encoding the sample clip")


def apply_vfr(source: Path, destination: Path, jitter_seconds: float = 0.05) -> None:
    """Re-encode with jittered presentation timestamps."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"setpts=PTS+random(1)*{jitter_seconds}/TB",
            "-fps_mode", "vfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(destination),
        ],
        check=True,
    )


def apply_rotation(source: Path, destination: Path, degrees: int) -> None:
    """Attach a display matrix without touching the pixels.

    The rotation is applied to the input side and the stream copied, so the
    output carries the flag exactly as a phone recording would.
    """
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-display_rotation", str(degrees),
            "-i", str(source),
            "-c", "copy",
            str(destination),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="photo containing a person")
    parser.add_argument("--out", type=Path, required=True, help="output clip path")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="attach a rotation flag, as phone video carries")
    parser.add_argument("--vfr", action="store_true", help="jitter frame timestamps")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        current = tmp_dir / "base.mp4"
        render_clip(args.image, current, args.width, args.height, args.fps, args.seconds)

        if args.vfr:
            nxt = tmp_dir / "vfr.mp4"
            apply_vfr(current, nxt)
            current = nxt
        if args.rotate:
            nxt = tmp_dir / "rot.mp4"
            apply_rotation(current, nxt, args.rotate)
            current = nxt

        args.out.write_bytes(current.read_bytes())

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
