"""MediaPipe Tasks PoseLandmarker implementation of PoseEstimator."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from app.pose.base import Landmark, PersonPose, PoseEstimator, RunningMode

_MP_RUNNING_MODES = {
    RunningMode.IMAGE: vision.RunningMode.IMAGE,
    RunningMode.VIDEO: vision.RunningMode.VIDEO,
}

LANDMARK_NAMES: tuple[str, ...] = tuple(landmark.name.lower() for landmark in vision.PoseLandmark)

# Skeleton topology, as (start, end) landmark index pairs. Sent to the client
# so drawing code does not have to hardcode one model's joint layout.
LANDMARK_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (connection.start, connection.end)
    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS
)


class MediaPipePoseEstimator(PoseEstimator):
    """Wraps PoseLandmarker.

    Multi-person detection is on by default: the user has to pick their
    climber out of whoever else is in the frame, so all of them have to be
    detected first.
    """

    def __init__(
        self,
        model_file: Path,
        *,
        mode: RunningMode = RunningMode.VIDEO,
        num_poses: int = 5,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mode = mode
        self._last_timestamp_ms = -1
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=_MP_RUNNING_MODES[mode],
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return LANDMARK_NAMES

    @property
    def landmark_connections(self) -> tuple[tuple[int, int], ...]:
        return LANDMARK_CONNECTIONS

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> tuple[PersonPose, ...]:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        if self._mode is RunningMode.VIDEO:
            # MediaPipe rejects a timestamp that does not advance, and a 30fps
            # clip rounds to whole milliseconds unevenly, so nudge forward
            # rather than letting a duplicate raise.
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self._landmarker.detect(image)
        return _to_people(result)

    def close(self) -> None:
        self._landmarker.close()


def _to_people(result: vision.PoseLandmarkerResult) -> tuple[PersonPose, ...]:
    people = []
    for landmarks in result.pose_landmarks or []:
        people.append(
            PersonPose(
                landmarks=tuple(
                    Landmark(
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        visibility=float(lm.visibility or 0.0),
                        presence=float(lm.presence or 0.0),
                    )
                    for lm in landmarks
                )
            )
        )
    return tuple(people)
