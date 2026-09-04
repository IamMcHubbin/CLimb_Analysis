"""Taking delivery of an upload.

Deliberately does no work beyond writing bytes and recording a row: the
transcode happens later, on the worker, so the HTTP request is not held open
for it. See app.ingest.job.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.db.repository import VideoRecord, VideoRepository, VideoStatus

NORMALISED_FILENAME = "normalised.mp4"
UPLOAD_FILENAME = "upload.bin"


class IngestService:
    """Creates the pending video row an ingest job will later fill in."""

    def __init__(
        self,
        repository: VideoRepository,
        settings: Settings = default_settings,
    ) -> None:
        self._repository = repository
        self._settings = settings

    def new_video_id(self) -> str:
        return uuid.uuid4().hex

    def upload_destination(self, video_id: str) -> Path:
        """Where the raw upload is streamed to.

        Inside the video's own directory, so that deleting that directory - on
        failure, or when retention comes round - takes the upload with it.
        """
        return self._settings.videos_dir / video_id / UPLOAD_FILENAME

    def register(self, video_id: str, original_filename: str) -> VideoRecord:
        """Record a staged upload as a pending video."""
        video_dir = self._settings.videos_dir / video_id
        record = VideoRecord(
            id=video_id,
            original_filename=original_filename,
            created_at=datetime.now(timezone.utc),
            status=VideoStatus.PENDING,
            # Known in advance because it is derived from the id, which keeps
            # this column non-null and the retention sweep simple.
            stored_path=self._settings.relative(video_dir / NORMALISED_FILENAME),
            upload_path=self._settings.relative(self.upload_destination(video_id)),
        )
        try:
            return self._repository.add(record)
        except Exception:
            shutil.rmtree(video_dir, ignore_errors=True)
            raise

    def discard(self, video_id: str) -> None:
        """Remove a staged upload that never became a video."""
        shutil.rmtree(self._settings.videos_dir / video_id, ignore_errors=True)
