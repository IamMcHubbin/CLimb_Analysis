"""Pose estimation.

``create_pose_estimator`` is the only place that names a concrete model, so
swapping MediaPipe for something else happens here.
"""

from __future__ import annotations

from app.config import Settings, settings as default_settings
from app.geometry import BoundingBox
from app.pose.base import (
    FramePose,
    Landmark,
    PersonPose,
    PoseEstimator,
    RunningMode,
)
from app.pose.model_zoo import MODEL_URLS, ensure_model, model_path


def create_pose_estimator(
    *,
    mode: RunningMode = RunningMode.VIDEO,
    variant: str | None = None,
    num_poses: int | None = None,
    settings: Settings = default_settings,
) -> PoseEstimator:
    chosen = variant or settings.pose_model
    count = num_poses if num_poses is not None else settings.max_people
    if chosen == "vitpose":
        # Optional experiment. Hugging Face owns these weights and its cache;
        # MediaPipe's .task downloader does not apply to this backend.
        from app.pose.vitpose_pose import VitPoseEstimator

        return VitPoseEstimator(mode=mode, num_poses=count)

    from app.pose.mediapipe_pose import MediaPipePoseEstimator

    return MediaPipePoseEstimator(
        ensure_model(chosen, settings=settings),
        mode=mode,
        num_poses=count,
    )


__all__ = [
    "BoundingBox",
    "FramePose",
    "Landmark",
    "MODEL_URLS",
    "PersonPose",
    "PoseEstimator",
    "RunningMode",
    "create_pose_estimator",
    "ensure_model",
    "model_path",
]
