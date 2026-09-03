"""The queue interface.

The queue carries nothing but a job id. Everything about a job - its status,
its progress, why it failed - lives in the database, which is what lets a page
refresh see the truth, and what will let a real broker replace the thread here
without anything else changing. Knowing nothing about job records is the point:
this interface is the same whether the backing store is SQLite or anything else.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

# Takes a job id and runs it to completion, updating job state as it goes.
JobHandler = Callable[[str], None]


class JobQueue(abc.ABC):
    """Somewhere to put job ids so they get run, one at a time, in order."""

    @abc.abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Submit a job for execution."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin consuming. Idempotent."""

    @abc.abstractmethod
    def stop(self, timeout: float | None = None) -> None:
        """Stop consuming and wait for the in-flight job to finish."""

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        """How many jobs are waiting, not counting one in flight."""
