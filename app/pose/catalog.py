"""Selectable backends, without importing or downloading their model runtimes."""

from __future__ import annotations

from importlib.util import find_spec

# Shown in the picker. ViTPose carries its cost in the label because the
# difference is not a nuance: measured on the gym clip it is roughly nine
# times Heavy per frame, which turned a two and a half minute analysis into
# seventeen minutes. Without a number there, choosing it looks like a hang.
# See docs/VITPOSE_EXPERIMENT.md.
MODEL_LABELS = {
    "lite": "MediaPipe Lite",
    "full": "MediaPipe Full",
    "heavy": "MediaPipe Heavy",
    "vitpose": "ViTPose+ Base (experimental, ~9x slower than Heavy on CPU)",
}

_VITPOSE_REQUIREMENTS = ("torch", "torchvision", "transformers", "scipy")


class ModelUnavailable(ValueError):
    pass


def unavailable_reason(model: str) -> str | None:
    """Why this model cannot run here, or None if it can.

    Uses ``find_spec`` rather than importing: this is called to render the
    picker, and importing torch to decide whether to offer it would cost
    seconds on every request.
    """
    if model == "vitpose" and any(find_spec(name) is None for name in _VITPOSE_REQUIREMENTS):
        return "Install requirements-vitpose.txt with CPU PyTorch, or build with INSTALL_VITPOSE=true."
    return None


def validate_model(model: str) -> str:
    if model not in MODEL_LABELS:
        raise ValueError(f"unknown pose model; choose one of {', '.join(MODEL_LABELS)}")
    reason = unavailable_reason(model)
    if reason:
        raise ModelUnavailable(reason)
    return model


def model_choices() -> list[dict]:
    return [
        {
            "id": name,
            "label": label,
            "available": unavailable_reason(name) is None,
            "unavailable_reason": unavailable_reason(name),
        }
        for name, label in MODEL_LABELS.items()
    ]
