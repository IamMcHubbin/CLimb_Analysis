"""Engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, settings as default_settings
from app.db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _create_engine(settings: Settings) -> Engine:
    url = settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        # The background analysis worker touches the DB from another thread.
        connect_args["check_same_thread"] = False
    engine = create_engine(url, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            # WAL lets the worker write job progress while a request reads it.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def init_db(settings: Settings = default_settings) -> None:
    """Create the engine, session factory and schema. Idempotent."""
    global _engine, _session_factory
    settings.ensure_dirs()
    if _engine is None:
        _engine = _create_engine(settings)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(_engine)
    # The project previously had no analysis_runs table or job foreign key.
    # Keep existing SQLite installations usable with this small additive
    # migration; a formal migration tool can replace this as the schema grows.
    if "jobs" in inspect(_engine).get_table_names():
        columns = {column["name"] for column in inspect(_engine).get_columns("jobs")}
        if "analysis_run_id" not in columns:
            with _engine.begin() as connection:
                connection.execute(text("ALTER TABLE jobs ADD COLUMN analysis_run_id VARCHAR(32)"))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_jobs_analysis_run_id ON jobs (analysis_run_id)"
                ))


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        raise RuntimeError("init_db() must be called before sessions are requested")
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine. Used by tests that point at a fresh directory."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
