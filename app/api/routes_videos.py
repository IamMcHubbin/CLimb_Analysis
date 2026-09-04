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
    get_analysis_run_repository,
    get_ingest_service,
    get_job_service,
    get_keypoint_store,
    get_settings,
    get_video_repository,
)
from app.api.schemas import AnalyseRequest, CandidatesOut, JobOut, KeypointsOut, VideoOut
from app.candidates import CandidateService
from app.config import Settings
from app.db.repository import AnalysisRunRepository, VideoRecord, VideoRepository
from app.frames import FrameReadError
from app.ingest.errors import IngestError, NormalisationFailed, UnreadableVideo
from app.ingest.service import IngestService
from app.jobs.service import JobService, UnknownCandidate
from app.keypoints import KeypointStore
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
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


@router.post("/{video_id}/analyse", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def analyse_video(
    video_id: str,
    request: AnalyseRequest,
    repository: VideoRepository = Depends(get_video_repository),
    service: JobService = Depends(get_job_service),
) -> JobOut:
    """Queue analysis of one candidate and return immediately.

    Poll GET /jobs/{id} for progress. Analysis takes roughly as long as the
    clip itself, so nothing useful can be returned synchronously.
    """
    _require_video(repository, video_id)
    try:
        job = service.submit(video_id, request.candidate_index)
    except UnknownCandidate as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return JobOut.from_record(job)


@router.get("/{video_id}/keypoints", response_model=KeypointsOut)
def get_keypoints(
    video_id: str,
    repository: VideoRepository = Depends(get_video_repository),
    store: KeypointStore = Depends(get_keypoint_store),
    analysis_run_id: str | None = Query(None),
    runs: AnalysisRunRepository = Depends(get_analysis_run_repository),
) -> KeypointsOut:
    """The finished track, index-aligned with the video's frames."""
    record = _require_video(repository, video_id)
    path = record.keypoints_path
    if analysis_run_id is not None:
        run = runs.get(analysis_run_id)
        if run is None or run.video_id != video_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown analysis run")
        path = run.keypoints_path
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this video has not been analysed yet",
        )
    data = store.read(path)

    # Rounded to four decimals: that is a fifth of a pixel at 1280 wide, and it
    # roughly halves the size of the response for a long clip.
    frames: list[list[list[float]] | None] = []
    match_iou: list[float | None] = []
    for index in range(data.metadata.frame_count):
        landmarks = data.frames.get(index)
        if landmarks is None:
            frames.append(None)
            match_iou.append(None)
            continue
        match_iou.append(round(data.ious.get(index, 0.0), 3))
        frames.append(
            [
                [round(landmark.x, 4), round(landmark.y, 4), round(landmark.visibility, 3)]
                for landmark in landmarks
            ]
        )

    return KeypointsOut(
        video_id=record.id,
        analysis_run_id=analysis_run_id or data.metadata.analysis_run_id,
        fps=data.metadata.fps,
        frame_count=data.metadata.frame_count,
        landmark_names=list(data.metadata.landmark_names),
        landmark_connections=[list(pair) for pair in data.metadata.landmark_connections],
        pose_model=data.metadata.pose_model,
        min_iou=data.metadata.min_iou,
        tracked_frame_count=data.tracked_frame_count,
        gap_frame_count=data.gap_frame_count,
        frames=frames,
        match_iou=match_iou,
    )
