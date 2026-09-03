"""Application settings, read from the environment once at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Everything the app needs to know about its environment.

    Paths under ``data_dir`` are the only writable locations. Records in the
    database store paths *relative* to ``data_dir`` so the directory can be
    moved or mounted elsewhere without rewriting rows.
    """

    data_dir: Path = field(default_factory=lambda: _env_path("CLIMB_DATA_DIR", Path.cwd() / "data"))
    model_dir: Path = field(default_factory=lambda: _env_path("CLIMB_MODEL_DIR", Path.cwd() / "models"))

    ffmpeg_bin: str = os.environ.get("CLIMB_FFMPEG", "ffmpeg")
    ffprobe_bin: str = os.environ.get("CLIMB_FFPROBE", "ffprobe")

    # Ingest normalisation targets. Every stored video matches these exactly.
    target_fps: int = _env_int("CLIMB_TARGET_FPS", 30)
    max_long_edge: int = _env_int("CLIMB_MAX_LONG_EDGE", 1280)

    max_upload_bytes: int = _env_int("CLIMB_MAX_UPLOAD_BYTES", 1024 * 1024 * 1024)

    # Set to point somewhere other than the SQLite file under data_dir.
    database_url_override: str | None = field(
        default_factory=lambda: os.environ.get("CLIMB_DATABASE_URL")
    )

    # Pose model variant: lite | full | heavy.
    pose_model: str = os.environ.get("CLIMB_POSE_MODEL", "lite")
    max_people: int = _env_int("CLIMB_MAX_PEOPLE", 5)

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def keypoints_dir(self) -> Path:
        return self.data_dir / "keypoints"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def database_url(self) -> str:
        return self.database_url_override or f"sqlite:///{self.data_dir / 'climb.db'}"

    def resolve(self, relative_path: str) -> Path:
        """Turn a DB-stored relative path into an absolute one."""
        return self.data_dir / relative_path

    def relative(self, path: Path) -> str:
        """Turn an absolute path under ``data_dir`` into its stored form."""
        return str(path.resolve().relative_to(self.data_dir))

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.videos_dir, self.keypoints_dir, self.uploads_dir, self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
