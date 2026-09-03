"""Request-scoped wiring.

Routes ask for a ``VideoRepository``; which implementation they get is decided
here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, settings as app_settings
from app.db.repository import VideoRepository
from app.db.session import session_scope
from app.db.sqlalchemy_repository import SqlAlchemyVideoRepository
from app.ingest.service import IngestService


def get_settings() -> Settings:
    return app_settings


def get_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session


def get_video_repository(session: Session = Depends(get_session)) -> VideoRepository:
    return SqlAlchemyVideoRepository(session)


def get_ingest_service(
    repository: VideoRepository = Depends(get_video_repository),
    settings: Settings = Depends(get_settings),
) -> IngestService:
    return IngestService(repository, settings=settings)
