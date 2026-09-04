from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db import init_db, reset_engine, session_scope  # noqa: E402
from app.db.repository import JobKind, JobRecord, JobStatus  # noqa: E402
from app.db.sqlalchemy_repository import (  # noqa: E402
    SqlAlchemyJobRepository,
    SqlAlchemyVideoRepository,
)
from app.ingest.job import IngestJobHandler  # noqa: E402
from app.ingest.service import IngestService  # noqa: E402
from app.jobs.base import JobQueue  # noqa: E402
from app.jobs.dispatch import JobDispatcher  # noqa: E402


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


@pytest.fixture
def ingest_video(settings, make_video):
    """Upload and normalise a clip, synchronously, and return the ready video.

    Runs the real ingest job rather than reaching past it, so tests get the
    same VideoRecord the application would produce.
    """

    def _ingest(filename: str = "clip.mp4", **clip) -> object:
        source = make_video(**clip)
        with session_scope() as session:
            service = IngestService(SqlAlchemyVideoRepository(session), settings=settings)
            video_id = service.new_video_id()
            destination = service.upload_destination(video_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            service.register(video_id, filename)
            job = SqlAlchemyJobRepository(session).add(
                JobRecord(
                    id=uuid.uuid4().hex,
                    video_id=video_id,
                    kind=JobKind.INGEST,
                    candidate_index=None,
                    status=JobStatus.QUEUED,
                    progress=0.0,
                    created_at=datetime.now(timezone.utc),
                )
            )
        # Committed above; the handler opens its own session, as it does in
        # production where it runs on the worker thread.
        IngestJobHandler(settings)(job.id)
        with session_scope() as session:
            return SqlAlchemyVideoRepository(session).get(video_id)

    return _ingest


class InlineJobQueue(JobQueue):
    """Runs each job the moment it is enqueued, on the calling thread.

    Lets API tests assert on finished state without threads or polling, while
    still going through the real dispatcher and handlers - and still only
    seeing jobs whose rows were committed first.
    """

    def __init__(self, settings: Settings, handlers) -> None:
        self._dispatch = JobDispatcher(handlers)
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)
        self._dispatch(job_id)

    def start(self) -> None:
        pass

    def stop(self, timeout: float | None = None) -> None:
        pass

    @property
    def depth(self) -> int:
        return 0
