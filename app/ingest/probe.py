"""ffprobe wrappers.

Everything downstream assumes the numbers here are true, so they are read from
the container rather than guessed, and the frame count of a normalised file is
obtained by actually decoding it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.ingest.errors import UnreadableVideo


@dataclass(frozen=True)
class VideoProbe:
    """What ffprobe says about a file.

    ``width``/``height`` are the *coded* dimensions. ``display_width``/
    ``display_height`` account for the rotation flag, i.e. what a correct
    player shows and what OpenCV does not.
    """

    width: int
    height: int
    rotation: int
    fps: float
    avg_fps: float
    duration_seconds: float
    codec: str
    nb_frames: int | None
    variable_frame_rate: bool

    @property
    def swaps_axes(self) -> bool:
        return self.rotation in (90, 270)

    @property
    def display_width(self) -> int:
        return self.height if self.swaps_axes else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.swaps_axes else self.height



def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise UnreadableVideo(proc.stderr.strip() or f"ffprobe failed: {' '.join(args)}")
    return proc.stdout


def _parse_rate(value: str | None) -> float:
    """ffprobe reports rates as rationals like '30000/1001'."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _extract_rotation(stream: dict) -> int:
    """Normalise the rotation flag to a clockwise angle in {0, 90, 180, 270}.

    It can arrive either as a display matrix in the side data (modern muxers)
    or as a legacy ``rotate`` tag. ffprobe reports the display matrix as a
    counter-clockwise angle, so the sign is flipped here.
    """
    angle: float | None = None
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            angle = -float(side_data["rotation"])
            break
    if angle is None:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                angle = float(tag)
            except ValueError:
                angle = None
    if angle is None:
        return 0
    return int(round(angle / 90.0) * 90) % 360


def detect_variable_frame_rate(
    path: Path,
    settings: Settings = default_settings,
    sample_seconds: int = 5,
) -> bool:
    """Decide whether a file is variable frame rate by looking at frame timing.

    Comparing the nominal and average rates is not good enough: a phone clip
    can average close to 30 while individual frames sit milliseconds apart.
    So the first few seconds are decoded and the gaps between presentation
    timestamps compared. A constant-rate file has identical gaps, whatever the
    rate, so 29.97 does not read as variable.

    Diagnostic only - normalisation retimes the file either way.
    """
    try:
        output = _run(
            [
                settings.ffprobe_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-read_intervals", f"%+{sample_seconds}",
                "-show_entries", "frame=pts_time",
                "-print_format", "json",
                str(path),
            ]
        )
    except UnreadableVideo:
        return False

    times = []
    for frame in json.loads(output or "{}").get("frames") or []:
        value = frame.get("pts_time")
        if value not in (None, "N/A"):
            times.append(float(value))
    times.sort()
    if len(times) < 10:
        return False

    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(gaps) < 8:
        return False
    gaps.sort()
    median = gaps[len(gaps) // 2]
    if median <= 0:
        return False
    return (gaps[-1] - gaps[0]) / median > 0.2


def probe_video(path: Path, settings: Settings = default_settings) -> VideoProbe:
    """Read stream metadata. Raises UnreadableVideo if there is no video stream."""
    output = _run(
        [
            settings.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "v:0",
            str(path),
        ]
    )
    payload = json.loads(output or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise UnreadableVideo(f"no video stream in {path.name}")
    stream = streams[0]

    if not stream.get("width") or not stream.get("height"):
        raise UnreadableVideo(f"video stream in {path.name} has no dimensions")

    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    nb_frames = stream.get("nb_frames")

    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        rotation=_extract_rotation(stream),
        fps=_parse_rate(stream.get("r_frame_rate")),
        avg_fps=_parse_rate(stream.get("avg_frame_rate")),
        duration_seconds=float(duration) if duration not in (None, "N/A") else 0.0,
        codec=stream.get("codec_name") or "unknown",
        nb_frames=int(nb_frames) if nb_frames not in (None, "N/A") else None,
        variable_frame_rate=detect_variable_frame_rate(path, settings=settings),
    )


def count_frames(path: Path, settings: Settings = default_settings) -> int:
    """Decode the file and count video frames.

    Container-level frame counts are not always present or correct, and the
    frontend maps playback time to a frame index, so this one number has to be
    exact. It costs a full decode pass - only run it on normalised files.
    """
    output = _run(
        [
            settings.ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-print_format", "json",
            str(path),
        ]
    )
    streams = json.loads(output or "{}").get("streams") or []
    if not streams or streams[0].get("nb_read_frames") in (None, "N/A"):
        raise UnreadableVideo(f"could not count frames in {path.name}")
    return int(streams[0]["nb_read_frames"])
