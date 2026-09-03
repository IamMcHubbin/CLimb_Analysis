"""A job queue that is one background thread and a list.

Deliberately the smallest thing that satisfies the interface. Jobs run one at
a time because pose estimation already saturates the CPU, so a second worker
would make both slower rather than finishing anything sooner.
"""

from __future__ import annotations

import logging
import queue
import threading

from app.jobs.base import JobHandler, JobQueue

logger = logging.getLogger(__name__)

_SHUTDOWN = object()


class ThreadedJobQueue(JobQueue):
    def __init__(self, handler: JobHandler, name: str = "analysis-worker") -> None:
        self._handler = handler
        self._name = name
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Not a daemon: stop() is called from the app's shutdown so an
            # in-flight job gets the chance to record why it stopped, rather
            # than leaving a row stuck at "running" forever.
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=False)
            self._thread.start()
            logger.info("job worker started")

    def stop(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._queue.put(_SHUTDOWN)
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("job worker did not stop within %ss", timeout)
        else:
            logger.info("job worker stopped")

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                # A handler that raises has already failed to record its own
                # failure, so log loudly - but never let it kill the worker,
                # or every later job silently stops running.
                try:
                    self._handler(item)
                except Exception:
                    logger.exception("job %s raised out of its handler", item)
            finally:
                self._queue.task_done()
