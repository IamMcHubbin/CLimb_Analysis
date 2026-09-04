"""Deleting footage once it is no longer needed.

The clip is the sensitive, bulky part and it stops being useful once the track
has been extracted. The keypoints and metadata are kept - they are small, they
are what the analysis was for, and they are not video of anybody.

Two windows, because the two cases differ. An analysed video has served its
purpose and is counted from when its analysis finished; the delay exists only
so the overlay has something to draw on while somebody reviews it. An upload
nobody ever analysed is abandoned, and counted from when it arrived.
"""

from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime, timedelta, timezone

from app.config import Settings, settings as default_settings
from app.db.repository import VideoRecord, VideoRepository
from app.db.session import session_scope
from app.db.sqlalchemy_repository import SqlAlchemyVideoRepository

logger = logging.getLogger(__name__)


class FootageRetention:
    """Applies the retention policy. Knows nothing about threads."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self._settings = settings

    def delete_footage(self, repository: VideoRepository, video: VideoRecord) -> bool:
        """Remove one video's files. Returns False if they were already gone.

        The whole per-video directory goes, which takes the cached candidate
        frame with it - that is a still of the same footage.
        """
        if not video.has_footage:
            return False
        video_dir = self._settings.resolve(video.stored_path).parent
        shutil.rmtree(video_dir, ignore_errors=True)
        repository.mark_footage_deleted(video.id)
        logger.info("deleted footage for video %s", video.id)
        return True

    def sweep(self, repository: VideoRepository, now: datetime | None = None) -> int:
        """Delete every video whose footage is past its window."""
        moment = now or datetime.now(timezone.utc)
        expired = repository.list_footage_to_delete(
            analysed_before=moment - timedelta(seconds=self._settings.retain_analysed_seconds),
            unanalysed_before=moment - timedelta(seconds=self._settings.retain_unanalysed_seconds),
        )
        return sum(1 for video in expired if self.delete_footage(repository, video))

    def expires_at(self, video: VideoRecord) -> datetime | None:
        """When this video's footage is due to go, or None if it already has."""
        if not video.has_footage:
            return None
        if video.analysis_completed_at is not None:
            return video.analysis_completed_at + timedelta(
                seconds=self._settings.retain_analysed_seconds
            )
        return video.created_at + timedelta(seconds=self._settings.retain_unanalysed_seconds)


class RetentionJanitor:
    """Runs the sweep on a timer.

    A timer rather than a check on access: the promise is that footage is
    deleted, not that it is hidden from whoever asks next. Nobody may ever ask
    again, and it still has to go.
    """

    def __init__(
        self,
        retention: FootageRetention,
        interval_seconds: int,
        name: str = "retention-janitor",
    ) -> None:
        self._retention = retention
        self._interval = max(1, interval_seconds)
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("retention janitor started (every %ss)", self._interval)

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    def sweep_once(self) -> int:
        with session_scope() as session:
            return self._retention.sweep(SqlAlchemyVideoRepository(session))

    def _run(self) -> None:
        # Sweep on startup too: the process may have been down for longer than
        # any retention window.
        while True:
            try:
                self.sweep_once()
            except Exception:
                # A failed sweep must not end the janitor, or footage stops
                # being deleted silently for the life of the process.
                logger.exception("retention sweep failed")
            if self._stop.wait(self._interval):
                return
