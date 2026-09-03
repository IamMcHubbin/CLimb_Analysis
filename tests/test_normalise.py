"""Ingest normalisation: the guarantees everything downstream relies on."""

from __future__ import annotations

import pytest

from app.ingest.normalise import build_ffmpeg_command, compute_target_size, normalise_video
from app.ingest.probe import probe_video


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1920, 1080, (1280, 720)),      # landscape, downscaled
        (1080, 1920, (720, 1280)),      # portrait, downscaled
        (640, 480, (640, 480)),         # already small, left alone
        (1280, 720, (1280, 720)),       # exactly at the cap
        (4000, 3000, (1280, 960)),      # 4:3
        (1001, 999, (1000, 998)),       # odd dimensions stepped down to even
    ],
)
def test_compute_target_size(width, height, expected):
    assert compute_target_size(width, height, 1280) == expected


def test_compute_target_size_never_upscales():
    assert compute_target_size(320, 240, 1280) == (320, 240)


def test_compute_target_size_rejects_nonsense():
    with pytest.raises(ValueError):
        compute_target_size(0, 100, 1280)


def test_ffmpeg_command_retimes_before_scaling():
    command = build_ffmpeg_command(
        __import__("pathlib").Path("in.mp4"),
        __import__("pathlib").Path("out.mp4"),
        width=1280,
        height=720,
        fps=30,
    )
    video_filter = command[command.index("-vf") + 1]
    assert video_filter.startswith("fps=30")
    assert "scale=1280:720" in video_filter
    # Stale rotation metadata must not survive into the output.
    assert "-map_metadata" in command


def test_rotation_is_baked_into_pixels(settings, make_video):
    """A portrait phone clip must come out physically upright."""
    source = make_video(width=640, height=480, rotate=90)
    source_probe = probe_video(source, settings=settings)
    assert source_probe.rotation != 0
    assert (source_probe.display_width, source_probe.display_height) == (480, 640)

    result = normalise_video(source, settings.videos_dir / "out.mp4", settings=settings)

    assert (result.width, result.height) == (480, 640)
    assert probe_video(result.path, settings=settings).rotation == 0


def test_variable_frame_rate_becomes_constant(settings, make_video):
    source = make_video(vfr=True, seconds=3.0)
    assert probe_video(source, settings=settings).variable_frame_rate is True

    result = normalise_video(source, settings.videos_dir / "out.mp4", settings=settings)

    output = probe_video(result.path, settings=settings)
    assert output.variable_frame_rate is False
    assert result.fps == settings.target_fps
    # Duration is derived from the true frame count, so the two must agree.
    assert result.duration_seconds == pytest.approx(result.frame_count / result.fps)


def test_long_edge_is_capped(settings, make_video):
    source = make_video(width=1920, height=1080, seconds=1.0)
    result = normalise_video(source, settings.videos_dir / "out.mp4", settings=settings)
    assert max(result.width, result.height) == settings.max_long_edge
    assert (result.width, result.height) == (1280, 720)


def test_frame_count_is_exact(settings, make_video):
    source = make_video(seconds=2.0, fps=30)
    result = normalise_video(source, settings.videos_dir / "out.mp4", settings=settings)
    assert result.frame_count == 60


def test_rejects_a_file_that_is_not_video(settings, tmp_path):
    from app.ingest.errors import UnreadableVideo

    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video")
    with pytest.raises(UnreadableVideo):
        normalise_video(junk, settings.videos_dir / "out.mp4", settings=settings)
