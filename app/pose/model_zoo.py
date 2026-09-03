"""Locating and fetching MediaPipe PoseLandmarker model files."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from app.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

# Accuracy and cost both increase down this list; the benchmark exists to show
# how much.
MODEL_URLS = {
    "lite": f"{_BASE_URL}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": f"{_BASE_URL}/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": f"{_BASE_URL}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


class UnknownModel(ValueError):
    pass


def model_path(variant: str, settings: Settings = default_settings) -> Path:
    if variant not in MODEL_URLS:
        raise UnknownModel(f"unknown pose model {variant!r}; expected one of {sorted(MODEL_URLS)}")
    return settings.model_dir / f"pose_landmarker_{variant}.task"


def ensure_model(variant: str, settings: Settings = default_settings) -> Path:
    """Return the local model file, downloading it once if missing.

    The container image downloads models at build time, so this is a
    convenience for running outside Docker rather than a runtime dependency.
    """
    destination = model_path(variant, settings=settings)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URLS[variant]
    logger.info("downloading pose model %s from %s", variant, url)
    # Download to a temporary name so an interrupted fetch is not mistaken for
    # a usable model on the next run.
    staging = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
            with staging.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"could not download pose model {variant!r}: {exc}") from exc
    staging.replace(destination)
    return destination
