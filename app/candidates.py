"""Finding the people in one frame so the user can point at the climber.

Detection runs on a single mid-clip frame rather than the whole video: it is
only there to let someone say "that one", and the middle of a clip is the most
likely place for the climber to be on the wall rather than walking up to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.db.repository import Candidate, CandidateSet, VideoRecord, VideoRepository
from app.frames import FrameReader, encode_jpeg
from app.pose import PoseEstimator, RunningMode, create_pose_estimator

logger = logging.getLogger(__name__)

CANDIDATE_FRAME_FILENAME = "candidate_frame.jpg"

EstimatorFactory = Callable[[], AbstractContextManager[PoseEstimator]]


class CandidateService:
    def __init__(
        self,
        repository: VideoRepository,
        settings: Settings = default_settings,
        estimator_factory: EstimatorFactory | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        # Injectable so the numbering and ordering rules can be tested without
        # loading a pose model.
        self._estimator_factory = estimator_factory or self._default_estimator

    def _default_estimator(self) -> AbstractContextManager[PoseEstimator]:
        # IMAGE mode: one frame, no temporal state to carry.
        return create_pose_estimator(mode=RunningMode.IMAGE, settings=self._settings)

    def frame_path(self, video: VideoRecord) -> Path:
        """Where the rendered candidate frame is cached."""
        return self._settings.resolve(video.stored_path).parent / CANDIDATE_FRAME_FILENAME

    def default_frame_index(self, video: VideoRecord) -> int:
        return max(0, video.frame_count // 2)

    def get_or_detect(self, video: VideoRecord, frame_index: int | None = None) -> CandidateSet:
        """Return the stored candidate set, detecting only if it is missing.

        Detection loads a pose model, which takes a second or two, and the
        client fetches the frame image straight after the JSON. Recomputing
        would also renumber the candidates underneath a user who is mid-choice.
        """
        wanted = self.default_frame_index(video) if frame_index is None else frame_index
        existing = self._repository.get_candidates(video.id)
        if existing is not None and existing.frame_index == wanted and self.frame_path(video).exists():
            return existing
        return self.detect(video, wanted)

    def detect(self, video: VideoRecord, frame_index: int | None = None) -> CandidateSet:
        """Detect people in one frame, cache the frame as JPEG, store the set."""
        wanted = self.default_frame_index(video) if frame_index is None else frame_index
        if video.frame_count <= 0:
            raise ValueError(f"video {video.id} has no frames")
        wanted = min(max(wanted, 0), video.frame_count - 1)

        video_path = self._settings.resolve(video.stored_path)
        with FrameReader(video_path) as reader:
            frame = reader.read_at(wanted)

        with self._estimator_factory() as estimator:
            people = estimator.detect(frame)

        # Ordered largest first, so the climber - usually the biggest figure -
        # tends to be candidate 0. The order is what the stored index refers
        # to, so it must not be recomputed differently later.
        ordered = sorted(people, key=lambda person: person.bounding_box.area, reverse=True)
        candidate_set = CandidateSet(
            frame_index=wanted,
            candidates=tuple(
                Candidate(
                    index=position,
                    bounding_box=person.bounding_box,
                    mean_visibility=person.mean_visibility,
                )
                for position, person in enumerate(ordered)
            ),
        )

        destination = self.frame_path(video)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encode_jpeg(frame))

        self._repository.save_candidates(video.id, candidate_set)
        logger.info(
            "detected %d candidate(s) in frame %d of video %s",
            len(candidate_set.candidates),
            wanted,
            video.id,
        )
        return candidate_set
