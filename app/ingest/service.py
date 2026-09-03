"""The ingest path: an arbitrary file in, a normalised video row out."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.db.repository import VideoRecord, VideoRepository
from app.ingest.normalise import normalise_video

NORMALISED_FILENAME = "normalised.mp4"


class IngestService:
    """Normalises an upload and records it.

    The original file is deleted once normalisation succeeds - it is never read
    again, and keeping it around invites code that reaches for it by mistake.
    """

    def __init__(
        self,
        repository: VideoRepository,
        settings: Settings = default_settings,
    ) -> None:
        self._repository = repository
        self._settings = settings

    def ingest(self, source_path: Path, original_filename: str) -> VideoRecord:
        video_id = uuid.uuid4().hex
        video_dir = self._settings.videos_dir / video_id
        destination = video_dir / NORMALISED_FILENAME

        try:
            normalised = normalise_video(source_path, destination, settings=self._settings)
        except Exception:
            shutil.rmtree(video_dir, ignore_errors=True)
            raise

        record = VideoRecord(
            id=video_id,
            original_filename=original_filename,
            created_at=datetime.now(timezone.utc),
            stored_path=self._settings.relative(normalised.path),
            width=normalised.width,
            height=normalised.height,
            fps=normalised.fps,
            frame_count=normalised.frame_count,
            duration_seconds=normalised.duration_seconds,
            size_bytes=normalised.size_bytes,
            source_width=normalised.source.display_width,
            source_height=normalised.source.display_height,
            source_fps=normalised.source.fps,
            source_rotation=normalised.source.rotation,
            source_codec=normalised.source.codec,
            source_variable_frame_rate=normalised.source.variable_frame_rate,
        )
        try:
            return self._repository.add(record)
        except Exception:
            shutil.rmtree(video_dir, ignore_errors=True)
            raise
