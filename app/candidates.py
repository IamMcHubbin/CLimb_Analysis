"""Finding the people in one frame so the user can point at the climber.

Detection runs on a single frame rather than the whole video: it is only there
to let someone say "that one". The middle of the clip is the first guess,
since that is where the climber is most likely to be on the wall rather than
walking up to it.

But one frame is a fragile thing to hang the whole flow on. On real gym
footage the middle frame turned out to be exactly where somebody walked
through the shot, and the picker came back with nobody in it and no way
forward. So when the caller has not asked for a specific frame, several are
tried, spread across the clip, and the first that finds anyone wins.
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

# Fractions of the clip to try, in order, when no frame was asked for. Spread
# out rather than clustered: an occlusion that hides the climber can easily
# last a couple of seconds, so neighbouring frames tend to fail together.
SEARCH_POSITIONS = (0.50, 0.40, 0.60, 0.30, 0.70, 0.20, 0.80, 0.10, 0.90)

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

    def _search_order(self, video: VideoRecord) -> list[int]:
        """Frame indices to try, in order, when the caller did not pick one."""
        last = max(0, video.frame_count - 1)
        order: list[int] = []
        for fraction in SEARCH_POSITIONS:
            index = min(last, max(0, int(video.frame_count * fraction)))
            if index not in order:
                order.append(index)
        return order

    def get_or_detect(self, video: VideoRecord, frame_index: int | None = None) -> CandidateSet:
        """Return the stored candidate set, detecting only if it is missing.

        Detection loads a pose model, which takes a second or two, and the
        client fetches the frame image straight after the JSON. Recomputing
        would also renumber the candidates underneath a user who is mid-choice.
        """
        existing = self._repository.get_candidates(video.id)
        if existing is not None and self.frame_path(video).exists():
            # A stored set for any frame satisfies an unspecified request; only
            # an explicit ask for a different frame forces a new detection.
            if frame_index is None or existing.frame_index == frame_index:
                return existing
        return self.detect(video, frame_index)

    def detect(self, video: VideoRecord, frame_index: int | None = None) -> CandidateSet:
        """Detect people, cache the frame as JPEG, store the numbered set.

        An explicit ``frame_index`` is honoured exactly, even if it finds
        nobody - the caller is steering, and silently answering about a
        different frame would be worse than an empty answer. Without one,
        frames across the clip are tried until somebody is found.
        """
        if video.frame_count <= 0:
            raise ValueError(f"video {video.id} has no frames")

        if frame_index is not None:
            attempts = [min(max(frame_index, 0), video.frame_count - 1)]
        else:
            attempts = self._search_order(video)

        video_path = self._settings.resolve(video.stored_path)
        with FrameReader(video_path) as reader, self._estimator_factory() as estimator:
            # The first attempt is kept as the fallback, so a video with nobody
            # in it still yields a frame the client can show and offer to
            # re-run somewhere else.
            wanted, frame, people = attempts[0], None, ()
            for attempt, index in enumerate(attempts):
                candidate_frame = reader.read_at(index)
                found = estimator.detect(candidate_frame)
                if attempt == 0:
                    frame = candidate_frame
                if found:
                    wanted, frame, people = index, candidate_frame, found
                    break
            if len(attempts) > 1 and not people:
                logger.info(
                    "no people found in video %s after trying %d frames",
                    video.id,
                    len(attempts),
                )

        # Ordered largest first. This is for stability, not because the
        # biggest box is the climber - on real gym footage it is the opposite,
        # a climber up the wall covers a tenth of the frame while someone
        # walking past covers two thirds. The order is what the stored index
        # refers to, so it must not be recomputed differently later.
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
