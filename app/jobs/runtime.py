"""The application's single job queue.

One module-level queue because there is one worker thread per process. Swapping
in a broker-backed JobQueue means changing what build_queue returns; nothing
that submits jobs needs to know.
"""

from __future__ import annotations

from app.analysis import AnalysisJobHandler
from app.config import Settings, settings as default_settings
from app.jobs.base import JobQueue
from app.jobs.inprocess import ThreadedJobQueue

_queue: JobQueue | None = None


def build_queue(settings: Settings = default_settings) -> JobQueue:
    return ThreadedJobQueue(AnalysisJobHandler(settings))


def get_queue() -> JobQueue:
    if _queue is None:
        raise RuntimeError("the job queue has not been started")
    return _queue


def set_queue(queue: JobQueue | None) -> None:
    """Install the process-wide queue. Called at startup, and by tests."""
    global _queue
    _queue = queue
