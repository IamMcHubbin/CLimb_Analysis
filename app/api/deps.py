"""Request-scoped wiring.

Routes ask for a ``VideoRepository``; which implementation they get is decided
here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, settings as app_settings
from app.db.repository import JobRepository, UnitOfWork, VideoRepository
from app.db.session import session_scope
from app.db.sqlalchemy_repository import (
    SqlAlchemyJobRepository,
    SqlAlchemyUnitOfWork,
    SqlAlchemyVideoRepository,
)
from app.candidates import CandidateService, EstimatorFactory
from app.ingest.service import IngestService
from app.jobs.base import JobQueue
from app.jobs.runtime import get_queue
from app.jobs.service import JobService
from app.keypoints import KeypointStore, ParquetKeypointStore
from app.retention import FootageRetention
from app.pose import RunningMode, create_pose_estimator


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


def get_estimator_factory(
    settings: Settings = Depends(get_settings),
) -> EstimatorFactory:
    """Builds a single-frame pose estimator.

    Its own dependency so a test can substitute a stub without also having to
    rebuild the service and its session-scoped repository.
    """

    def factory():
        return create_pose_estimator(mode=RunningMode.IMAGE, settings=settings)

    return factory


def get_candidate_service(
    repository: VideoRepository = Depends(get_video_repository),
    settings: Settings = Depends(get_settings),
    estimator_factory: EstimatorFactory = Depends(get_estimator_factory),
) -> CandidateService:
    return CandidateService(repository, settings=settings, estimator_factory=estimator_factory)


def get_job_repository(session: Session = Depends(get_session)) -> JobRepository:
    return SqlAlchemyJobRepository(session)


def get_job_queue() -> JobQueue:
    return get_queue()


def get_unit_of_work(session: Session = Depends(get_session)) -> UnitOfWork:
    return SqlAlchemyUnitOfWork(session)


def get_job_service(
    jobs: JobRepository = Depends(get_job_repository),
    videos: VideoRepository = Depends(get_video_repository),
    queue: JobQueue = Depends(get_job_queue),
    unit_of_work: UnitOfWork = Depends(get_unit_of_work),
) -> JobService:
    return JobService(jobs, videos, queue, unit_of_work)


def get_keypoint_store(settings: Settings = Depends(get_settings)) -> KeypointStore:
    return ParquetKeypointStore(settings)


def get_retention(settings: Settings = Depends(get_settings)) -> FootageRetention:
    return FootageRetention(settings)
