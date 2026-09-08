"""Experimental, stateless CPU ViTPose+ with RT-DETR person detection.

ViTPose is top-down: each call detects people across the entire frame, then
poses all retained boxes in one batch. Both running modes are independent
per-frame detections; the application's existing tracker handles identity.

COCO has 17 landmarks and no depth. ``z=0`` denotes unavailable depth, not a
3D estimate. Visibility/presence both carry the clipped heatmap confidence:
these are proxies, NOT calibrated MediaPipe visibility/presence probabilities.
No low-confidence joint is dropped (doing so would change the stored schema).

Weights use Transformers' normal from_pretrained cache. Revisions are pinned
because AnalysisRun currently records the variant, not individual checkpoints.
Install requirements-vitpose.txt only when opting into this experiment.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.pose.base import Landmark, PersonPose, PoseEstimator, RunningMode

POSE_MODEL = "usyd-community/vitpose-plus-base"
POSE_REVISION = "92be54d7a29e42fad47b6e2ca01dd9e685a61e0d"
DETECTOR_MODEL = "PekingU/rtdetr_r18vd"
DETECTOR_REVISION = "ac77a11ff0170a41b771c03264987f8ce2b0d753"
DETECTOR_THRESHOLD = 0.3
COCO_PERSON_LABEL = 0
COCO_DATASET_INDEX = 0

LANDMARK_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LANDMARK_CONNECTIONS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


def _person_boxes(result, width: int, height: int, limit: int) -> np.ndarray:
    """Keep person detections in confidence order, return pixel COCO xywh."""
    boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(result["scores"], dtype=np.float32)
    labels = np.asarray(result["labels"])
    keep = (labels == COCO_PERSON_LABEL) & (scores >= DETECTOR_THRESHOLD)
    keep &= np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    boxes, scores = boxes[keep].copy(), scores[keep]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    boxes[:, 2:] -= boxes[:, :2]
    valid = (boxes[:, 2:] > 0).all(axis=1)
    boxes, scores = boxes[valid], scores[valid]
    return boxes[np.argsort(-scores, kind="stable")[:limit]]


def _to_person(result, width: int, height: int) -> PersonPose:
    """Postprocessed coordinates are full-frame pixels, never crop pixels."""
    points = np.asarray(result["keypoints"], dtype=np.float64)
    scores = np.asarray(result["scores"], dtype=np.float64)
    labels = np.asarray(result["labels"])
    count = len(LANDMARK_NAMES)
    if points.shape != (count, 2) or scores.shape != (count,):
        raise ValueError("ViTPose must return all 17 COCO landmarks")
    if labels.shape != (count,) or set(labels.tolist()) != set(range(count)):
        raise ValueError("ViTPose returned an unexpected COCO landmark schema")
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("ViTPose returned non-finite keypoints or confidence")
    order = np.argsort(labels)
    points = np.clip(points[order] / [width, height], 0.0, 1.0)
    scores = np.clip(scores[order], 0.0, 1.0)
    return PersonPose(tuple(
        Landmark(float(x), float(y), 0.0, float(score), float(score))
        for (x, y), score in zip(points, scores)
    ))


class VitPoseEstimator(PoseEstimator):
    """Multi-person RT-DETR-R18 + ViTPose+ Base, explicitly on CPU."""

    def __init__(self, *, mode: RunningMode = RunningMode.VIDEO, num_poses: int = 5):
        RunningMode(mode)  # Validate the contract; neither mode adds temporal state.
        if num_poses < 1:
            raise ValueError("num_poses must be positive")
        self._num_poses = num_poses
        self._closed = False
        try:
            import torch
            from transformers import (
                RTDetrForObjectDetection, RTDetrImageProcessor,
                VitPoseForPoseEstimation, VitPoseImageProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "The vitpose experiment needs optional dependencies; "
                "install requirements-vitpose.txt (see docs/VITPOSE_EXPERIMENT.md)."
            ) from exc
        self._torch = torch
        self._detector_processor = RTDetrImageProcessor.from_pretrained(
            DETECTOR_MODEL, revision=DETECTOR_REVISION,
        )
        self._detector = RTDetrForObjectDetection.from_pretrained(
            DETECTOR_MODEL, revision=DETECTOR_REVISION, use_safetensors=True,
        ).to("cpu").eval()
        self._pose_processor = VitPoseImageProcessor.from_pretrained(
            POSE_MODEL, revision=POSE_REVISION,
        )
        self._pose = VitPoseForPoseEstimation.from_pretrained(
            POSE_MODEL, revision=POSE_REVISION, use_safetensors=True,
        ).to("cpu").eval()

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return LANDMARK_NAMES

    @property
    def landmark_connections(self) -> tuple[tuple[int, int], ...]:
        return LANDMARK_CONNECTIONS

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> tuple[PersonPose, ...]:
        if self._closed:
            raise RuntimeError("ViTPose estimator is closed")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("expected a uint8 BGR image with three channels")
        height, width = frame_bgr.shape[:2]
        if not width or not height:
            raise ValueError("empty frame")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Explicit CPU tensors and inference mode; no timestamp/identity cache.
        with self._torch.inference_mode():
            inputs = self._detector_processor(images=rgb, return_tensors="pt")
            result = self._detector_processor.post_process_object_detection(
                self._detector(**inputs), target_sizes=[(height, width)],
                threshold=DETECTOR_THRESHOLD,
            )[0]
            boxes = _person_boxes(result, width, height, self._num_poses)
            if not len(boxes):
                return ()
            inputs = self._pose_processor(images=rgb, boxes=[boxes], return_tensors="pt")
            # One expert index per PERSON crop, not per original image.
            inputs["dataset_index"] = self._torch.full(
                (len(boxes),), COCO_DATASET_INDEX, dtype=self._torch.long, device="cpu",
            )
            results = self._pose_processor.post_process_pose_estimation(
                self._pose(**inputs), boxes=[boxes], threshold=None,
            )[0]
            if len(results) != len(boxes):
                raise ValueError("ViTPose did not return one pose per person box")
            return tuple(_to_person(result, width, height) for result in results)

    def close(self) -> None:
        self._closed = True
        self._pose = self._detector = None
        self._pose_processor = self._detector_processor = None
