"""Normalising an upload, as a queued job.

Normalisation used to happen inside the upload request, which meant a 50MB
clip held an HTTP connection open for the length of a transcode. Behind a
reverse proxy that is a timeout waiting to happen - Cloudflare gives up on an
origin after about 100 seconds - and even when it worked the user got a
spinner with nothing behind it.

So the request now only takes delivery of the bytes, and this runs afterwards
on the same worker thread as analysis. ffmpeg reports its own progress, so the
job can say how far along it is rather than just that it is busy.
"""

from __future__ import annotations

import logging

from app.config import Settings, settings as default_settings
from app.db.repository import VideoRecord, VideoStatus
from app.db.session import session_scope
from app.db.sqlalchemy_repository import SqlAlchemyJobRepository, SqlAlchemyVideoRepository
from app.ingest.errors import IngestError, UnreadableVideo
from app.ingest.normalise import normalise_video
from app.ingest.service import NORMALISED_FILENAME

logger = logging.getLogger(__name__)


class IngestJobHandler:
    """Turns one pending upload into a normalised video."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self._settings = settings

    def __call__(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except UnreadableVideo:
            logger.exception("ingest job %s: not a readable video", job_id)
            self._fail(job_id, "this file does not contain a video stream ffmpeg can read")
        except Exception as exc:
            logger.exception("ingest job %s failed", job_id)
            self._fail(job_id, self._sanitise(str(exc)))

    def _sanitise(self, message: str) -> str:
        """Keep server paths out of a message the client will be shown.

        ffmpeg reports errors with the full path of the file it choked on,
        which is a staged upload inside the data directory - of no use to the
        caller and not theirs to see.
        """
        cleaned = message.replace(str(self._settings.data_dir), "").strip()
        return cleaned or "this upload could not be normalised"

    def _fail(self, job_id: str, message: str) -> None:
        try:
            with session_scope() as session:
                jobs = SqlAlchemyJobRepository(session)
                job = jobs.get(job_id)
                jobs.mark_failed(job_id, message)
                if job is not None:
                    # The video is failed too, not left pending forever: the
                    # client polls the video, and a status that never changes
                    # is worse than a stated failure.
                    SqlAlchemyVideoRepository(session).mark_ingest_failed(job.video_id, message)
        except Exception:
            logger.exception("could not record the failure of ingest job %s", job_id)

    def _run(self, job_id: str) -> None:
        with session_scope() as session:
            jobs = SqlAlchemyJobRepository(session)
            job = jobs.get(job_id)
            if job is None:
                raise IngestError(f"job {job_id} no longer exists")
            video = SqlAlchemyVideoRepository(session).get(job.video_id)
            if video is None:
                raise IngestError(f"video {job.video_id} no longer exists")
            if video.status is not VideoStatus.PENDING:
                logger.info("ingest job %s: video already %s", job_id, video.status.value)
                jobs.mark_done(job_id)
                return
            if video.upload_path is None:
                raise IngestError("the uploaded file is gone")
            jobs.mark_running(job_id)
            upload_path = video.upload_path
            video_id = video.id
            original_filename = video.original_filename
            created_at = video.created_at
            stored_path = video.stored_path

        source = self._settings.resolve(upload_path)
        destination = self._settings.resolve(stored_path)
        if not source.exists():
            raise IngestError("the uploaded file is gone")

        last_reported = -1.0

        def report(fraction: float) -> None:
            # Written at most every percent: SQLite is not the place for an
            # update per progress line.
            nonlocal last_reported
            if fraction - last_reported < 0.01:
                return
            last_reported = fraction
            with session_scope() as session:
                SqlAlchemyJobRepository(session).update_progress(job_id, fraction)

        normalised = normalise_video(
            source, destination, settings=self._settings, on_progress=report
        )
        # The upload has served its purpose and is never read again.
        source.unlink(missing_ok=True)

        record = VideoRecord(
            id=video_id,
            original_filename=original_filename,
            created_at=created_at,
            status=VideoStatus.READY,
            stored_path=stored_path,
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
        with session_scope() as session:
            SqlAlchemyVideoRepository(session).mark_ready(video_id, record)
            SqlAlchemyJobRepository(session).mark_done(job_id)
        logger.info(
            "ingested video %s: %dx%d, %d frames",
            video_id, normalised.width, normalised.height, normalised.frame_count,
        )


__all__ = ["IngestJobHandler", "NORMALISED_FILENAME"]
