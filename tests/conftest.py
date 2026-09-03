from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import init_db, reset_engine  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a throwaway data directory, with a fresh engine.

    The engine is initialised here rather than left to the application
    lifespan: the lifespan uses the module-level settings built at import time,
    and init_db is idempotent, so claiming it first is what keeps a test off
    the real data directory.
    """
    reset_engine()
    configured = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    configured.ensure_dirs()
    init_db(configured)
    yield configured
    reset_engine()


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def make_video(tmp_path):
    """Build small synthetic clips: CFR, VFR, rotated, or any combination."""

    def _make(
        name: str = "clip.mp4",
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        seconds: float = 2.0,
        rotate: int = 0,
        vfr: bool = False,
    ) -> Path:
        base = tmp_path / f"base_{name}"
        _ffmpeg(
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(base),
        )
        current = base
        if vfr:
            jittered = tmp_path / f"vfr_{name}"
            _ffmpeg(
                "-i", str(current),
                "-vf", "setpts=PTS+random(1)*0.06/TB",
                "-fps_mode", "vfr",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(jittered),
            )
            current = jittered
        if rotate:
            rotated = tmp_path / f"rot_{name}"
            # Attaches a display matrix without touching the pixels, the way a
            # phone records a clip held sideways.
            _ffmpeg("-display_rotation", str(rotate), "-i", str(current), "-c", "copy", str(rotated))
            current = rotated
        return current

    return _make
