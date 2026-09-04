"""Routing a job id to the handler for its kind.

The queue carries ids and nothing else, so something has to look up what a job
actually is. That lookup lives here rather than in the queue, which keeps the
queue ignorant of job types and therefore still swappable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from app.db.repository import JobKind
from app.db.session import session_scope
from app.db.sqlalchemy_repository import SqlAlchemyJobRepository
from app.jobs.base import JobHandler

logger = logging.getLogger(__name__)


class JobDispatcher:
    def __init__(self, handlers: Mapping[JobKind, JobHandler]) -> None:
        self._handlers = dict(handlers)

    def __call__(self, job_id: str) -> None:
        with session_scope() as session:
            job = SqlAlchemyJobRepository(session).get(job_id)
        if job is None:
            logger.warning("job %s was queued but no longer exists", job_id)
            return

        handler = self._handlers.get(job.kind)
        if handler is None:
            message = f"no handler for job kind {job.kind.value!r}"
            logger.error("%s (job %s)", message, job_id)
            with session_scope() as session:
                SqlAlchemyJobRepository(session).mark_failed(job_id, message)
            return

        handler(job_id)
