"""Job status.

Answers from the database, not from the queue, so a page refresh or a restart
still sees the truth about a job.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_job_repository
from app.api.schemas import JobOut
from app.db.repository import JobRepository

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    repository: JobRepository = Depends(get_job_repository),
) -> JobOut:
    job = repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown job")
    return JobOut.from_record(job)
