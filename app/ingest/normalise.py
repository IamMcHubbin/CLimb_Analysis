"""Turn an arbitrary upload into a video the rest of the app can trust.

Three properties are guaranteed for every stored file:

* constant frame rate, so frame index == round(time * fps) holds exactly;
* rotation baked into the pixels, so OpenCV (which ignores the display matrix)
  sees the same orientation as the browser;
* long edge capped, so inference cost is bounded and predictable.

Phone footage violates all three. Nothing downstream re-checks, so this module
verifies its own output and fails loudly rather than passing on a bad file.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.ingest.errors import NormalisationFailed
from app.ingest.probe import VideoProbe, count_frames, probe_video


@dataclass(frozen=True)
class NormalisedVideo:
    """The output file plus the probe of the source it came from."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    size_bytes: int
    source: VideoProbe


def _even(value: float) -> int:
    """Round to an even integer. H.264 with yuv420p requires even dimensions.

    Rounds to nearest and then steps down to even, rather than rounding the
    halved value - Python's banker's rounding makes that inconsistent between
    neighbouring odd numbers.
    """
    rounded = int(round(value))
    return max(2, rounded - (rounded % 2))


def compute_target_size(width: int, height: int, max_long_edge: int) -> tuple[int, int]:
    """Cap the long edge, preserve aspect ratio, never upscale.

    Takes *display* dimensions (i.e. after rotation), so portrait and landscape
    sources are treated identically.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid source dimensions: {width}x{height}")
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return _even(width), _even(height)
    scale = max_long_edge / long_edge
    return _even(width * scale), _even(height * scale)


# Called with 0.0-1.0 as the transcode advances.
ProgressCallback = Callable[[float], None]


def build_ffmpeg_command(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
    fps: int,
    settings: Settings = default_settings,
) -> list[str]:
    """The normalisation command. Pure, so it can be asserted on in tests.

    Autorotation is ffmpeg's default and is what bakes the display matrix into
    the pixels; the target size is computed from post-rotation dimensions to
    match. ``-map_metadata -1`` then drops the stale rotate tag so the output
    cannot be rotated a second time by a player.
    """
    # fps before scale: retiming first means fewer frames to rescale.
    video_filter = f"fps={fps},scale={width}:{height}:flags=bicubic,setsar=1"
    return [
        settings.ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        # Machine-readable progress on stdout, so a long transcode can be
        # reported rather than shown as an indeterminate spinner.
        "-progress", "pipe:1",
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",           # audio and data streams are not used
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-map_metadata", "-1",
        "-movflags", "+faststart",     # so the browser can start playing early
        str(destination),
    ]


def _run_ffmpeg(
    command: list[str],
    duration_seconds: float,
    on_progress: ProgressCallback | None,
) -> None:
    """Run ffmpeg, reporting progress as it goes.

    stderr goes to a file rather than a pipe: reading stdout while stderr
    fills its own pipe buffer is the classic way to deadlock a subprocess.
    """
    with tempfile.TemporaryFile("w+") as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, text=True
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if on_progress is None or duration_seconds <= 0:
                    continue
                key, _, value = line.strip().partition("=")
                if key != "out_time_ms" or not value.isdigit():
                    continue
                fraction = (int(value) / 1_000_000) / duration_seconds
                on_progress(min(1.0, max(0.0, fraction)))
        finally:
            process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            errors.seek(0)
            raise NormalisationFailed(errors.read().strip() or "ffmpeg failed")


def normalise_video(
    source: Path,
    destination: Path,
    settings: Settings = default_settings,
    on_progress: ProgressCallback | None = None,
) -> NormalisedVideo:
    """Normalise ``source`` into ``destination`` and measure the result."""
    source_probe = probe_video(source, settings=settings)
    width, height = compute_target_size(
        source_probe.display_width,
        source_probe.display_height,
        settings.max_long_edge,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(
        source,
        destination,
        width=width,
        height=height,
        fps=settings.target_fps,
        settings=settings,
    )
    _run_ffmpeg(command, source_probe.duration_seconds, on_progress)
    if not destination.exists():
        raise NormalisationFailed("ffmpeg produced no output")

    output = probe_video(destination, settings=settings)
    if output.rotation != 0:
        raise NormalisationFailed(
            f"normalised file still carries a rotation flag ({output.rotation} degrees)"
        )
    if (output.width, output.height) != (width, height):
        raise NormalisationFailed(
            f"expected {width}x{height}, ffmpeg produced {output.width}x{output.height}"
        )
    if abs(output.fps - settings.target_fps) > 0.01:
        raise NormalisationFailed(
            f"expected {settings.target_fps}fps, ffmpeg produced {output.fps}"
        )
    if output.variable_frame_rate:
        raise NormalisationFailed("normalised file still has variable frame timing")

    frame_count = count_frames(destination, settings=settings)
    if frame_count <= 0:
        raise NormalisationFailed("normalised file contains no frames")
    return NormalisedVideo(
        path=destination,
        width=output.width,
        height=output.height,
        fps=float(settings.target_fps),
        frame_count=frame_count,
        # Derived from the true frame count, not the container duration: this
        # is the timeline the frontend scrubs against.
        duration_seconds=frame_count / float(settings.target_fps),
        size_bytes=destination.stat().st_size,
        source=source_probe,
    )
