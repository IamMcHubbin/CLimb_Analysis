"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_jobs import router as jobs_router
from app.api.routes_videos import router as videos_router
from app.config import settings
from app.db.session import init_db, session_scope
from app.db.sqlalchemy_repository import SqlAlchemyJobRepository
from app.jobs.runtime import build_queue, set_queue
from app.jobs.service import recover_unfinished_jobs
from app.retention import FootageRetention, RetentionJanitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


SHUTDOWN_GRACE_SECONDS = 30


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    init_db(settings)

    queue = build_queue(settings)
    set_queue(queue)
    # Before starting the worker: a job left running by the last process is
    # dead, and a job left queued never ran. Sorting that out first means the
    # worker never sees a job whose row disagrees with reality.
    with session_scope() as session:
        recover_unfinished_jobs(SqlAlchemyJobRepository(session), queue)
    queue.start()

    janitor = RetentionJanitor(
        FootageRetention(settings), settings.retention_sweep_seconds
    )
    janitor.start()
    try:
        yield
    finally:
        janitor.stop()
        # Give the in-flight job a chance to record how it ended rather than
        # leaving its row stuck at "running".
        queue.stop(timeout=SHUTDOWN_GRACE_SECONDS)
        set_queue(None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Climb Analysis",
        description="Pose analysis for climbing video (proof of concept).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(videos_router)
    app.include_router(jobs_router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config", tags=["ops"])
    def client_config() -> dict[str, object]:
        """Limits the client needs to know, so they are stated in one place.

        Without this the upload cap would be written down twice and drift.
        """
        return {
            "max_upload_bytes": settings.max_upload_bytes,
            "target_fps": settings.target_fps,
            "max_long_edge": settings.max_long_edge,
            "retain_analysed_seconds": settings.retain_analysed_seconds,
            "retain_unanalysed_seconds": settings.retain_unanalysed_seconds,
        }

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
