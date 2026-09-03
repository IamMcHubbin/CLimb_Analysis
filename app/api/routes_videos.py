"""Video routes: ingest, and picking which person in the frame is the climber.

Analysis jobs and keypoints come later.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import (
    get_candidate_service,
    get_ingest_service,
    get_settings,
    get_video_repository,
)
from app.api.schemas import CandidatesOut, VideoOut
from app.candidates import CandidateService
from app.config import Settings
from app.db.repository import VideoRecord, VideoRepository
from app.frames import FrameReadError
from app.ingest.errors import IngestError, NormalisationFailed, UnreadableVideo
from app.ingest.service import IngestService
from app.ingest.upload import EmptyUpload, UploadTooLarge, save_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        yield chunk


@router.post("", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    service: IngestService = Depends(get_ingest_service),
    settings: Settings = Depends(get_settings),
) -> VideoOut:
    """Accept a video, normalise it, and return its metadata.

    Normalisation is synchronous: it is an ffmpeg transcode, so a long clip
    will hold the request open. That is deliberate for now - nothing can be
    done with a video until it is normalised, and moving it into the job queue
    would mean a video row that exists but is not yet usable.
    """
    staged = settings.uploads_dir / f"{uuid.uuid4().hex}.upload"
    try:
        await save_stream(_chunks(file), staged, settings.max_upload_bytes)

        try:
            record = service.ingest(staged, file.filename or "upload")
        except UnreadableVideo as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"could not read a video stream from this file: {exc}",
            ) from exc
        except NormalisationFailed as exc:
            logger.exception("normalisation failed for %s", file.filename)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"could not normalise this video: {exc}",
            ) from exc
    except UploadTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except EmptyUpload as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except IngestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        staged.unlink(missing_ok=True)
        await file.close()

    return VideoOut.from_record(record)


@router.get("", response_model=list[VideoOut])
def list_videos(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    repository: VideoRepository = Depends(get_video_repository),
) -> list[VideoOut]:
    return [VideoOut.from_record(record) for record in repository.list(limit=limit, offset=offset)]


def _require_video(repository: VideoRepository, video_id: str) -> VideoRecord:
    record = repository.get(video_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown video")
    return record


@router.get("/{video_id}", response_model=VideoOut)
def get_video(
    video_id: str,
    repository: VideoRepository = Depends(get_video_repository),
) -> VideoOut:
    return VideoOut.from_record(_require_video(repository, video_id))


@router.get("/{video_id}/file")
def get_video_file(
    video_id: str,
    repository: VideoRepository = Depends(get_video_repository),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve the normalised file for playback.

    Returned via FileResponse so range requests work and the browser can seek.
    """
    record = _require_video(repository, video_id)
    path = settings.resolve(record.stored_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="video file is missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}.mp4")


@router.get("/{video_id}/candidates", response_model=CandidatesOut)
def get_candidates(
    video_id: str,
    frame_index: int | None = Query(
        None, ge=0, description="Frame to detect in. Defaults to the middle of the clip."
    ),
    refresh: bool = Query(False, description="Re-run detection even if a set is stored."),
    repository: VideoRepository = Depends(get_video_repository),
    service: CandidateService = Depends(get_candidate_service),
) -> CandidatesOut:
    """Detect everyone in one frame, so the client can ask which is the climber.

    The result is stored: the index returned here is what starts an analysis,
    and re-running detection would renumber the candidates underneath the user.
    Pass a different `frame_index` if the climber is not in shot mid-clip.
    """
    record = _require_video(repository, video_id)
    try:
        if refresh:
            candidate_set = service.detect(record, frame_index)
        else:
            candidate_set = service.get_or_detect(record, frame_index)
    except FrameReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"could not read that frame: {exc}",
        ) from exc
    return CandidatesOut.from_set(record, candidate_set)


@router.get("/{video_id}/candidates/frame.jpg")
def get_candidate_frame(
    video_id: str,
    repository: VideoRepository = Depends(get_video_repository),
    service: CandidateService = Depends(get_candidate_service),
) -> FileResponse:
    """The frame the candidates were detected in, for the client to draw on."""
    record = _require_video(repository, video_id)
    path = service.frame_path(record)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no candidate frame yet; call /candidates first",
        )
    return FileResponse(path, media_type="image/jpeg")
