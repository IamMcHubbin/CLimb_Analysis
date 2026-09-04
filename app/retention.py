"""Deleting footage once it is no longer needed.

The clip is the sensitive, bulky part and it stops being useful once the track
has been extracted. The keypoints and metadata are kept - they are small, they
are what the analysis was for, and they are not video of anybody.

Keypoint files are a separate matter. Each analysis run writes its own, so a
video can have several and none of them should go when its footage does. What
does go is an artifact no run points at any more: one left behind by a job
that died mid-write, or by a video whose rows have been deleted. Those are
swept; anything a run still references never is.

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
from app.db.repository import AnalysisRunRepository, VideoRecord, VideoRepository
from app.db.session import session_scope
from app.db.sqlalchemy_repository import (
    SqlAlchemyAnalysisRunRepository,
    SqlAlchemyVideoRepository,
)

logger = logging.getLogger(__name__)

# How long an unreferenced artifact is left alone before it counts as an
# orphan. Covers the window between a worker writing its file and committing
# the path onto its run.
ORPHAN_GRACE_SECONDS = 3600


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

    def sweep_orphan_artifacts(
        self,
        videos: VideoRepository,
        runs: AnalysisRunRepository,
        now: datetime | None = None,
    ) -> int:
        """Delete keypoint files no analysis run refers to any more.

        Referenced files are never touched, so a result somebody can still ask
        for cannot be swept out from under them. Files younger than the grace
        period are also left alone: a job writes its artifact before recording
        the path on its run, and that gap must not look like an orphan.
        """
        moment = now or datetime.now(timezone.utc)
        keypoints_dir = self._settings.keypoints_dir
        if not keypoints_dir.exists():
            return 0

        referenced = runs.referenced_keypoint_paths()
        # A video's "latest" pointer counts as a reference too, so an artifact
        # a client can still reach through the video is never an orphan.
        for video in videos.list(limit=10_000):
            if video.keypoints_path:
                referenced.add(video.keypoints_path)

        cutoff = moment - timedelta(seconds=ORPHAN_GRACE_SECONDS)
        removed = 0
        for path in keypoints_dir.glob("*.parquet"):
            # Skip the writer's own temporary files; they start with a dot and
            # belong to a write that has not finished.
            if path.name.startswith("."):
                continue
            if self._settings.relative(path) in referenced:
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified > cutoff:
                continue
            path.unlink(missing_ok=True)
            logger.info("deleted orphaned keypoint artifact %s", path.name)
            removed += 1
        return removed

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
        """Both sweeps: expired footage, then artifacts nothing refers to."""
        with session_scope() as session:
            videos = SqlAlchemyVideoRepository(session)
            runs = SqlAlchemyAnalysisRunRepository(session)
            return (
                self._retention.sweep(videos)
                + self._retention.sweep_orphan_artifacts(videos, runs)
            )

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
