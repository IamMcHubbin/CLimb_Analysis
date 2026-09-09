"""Selectable backends, without importing or downloading their model runtimes."""
from importlib.util import find_spec

MODEL_LABELS = {
    'lite': 'MediaPipe Lite',
    'full': 'MediaPipe Full',
    'heavy': 'MediaPipe Heavy',
    'vitpose': 'ViTPose+ Base (experimental, slow CPU)',
}


class ModelUnavailable(ValueError):
    pass


def unavailable_reason(model: str) -> str | None:
    if model == 'vitpose' and any(find_spec(name) is None for name in
                                  ('torch', 'torchvision', 'transformers', 'scipy')):
        return 'Install requirements-vitpose.txt with CPU PyTorch, or build with INSTALL_VITPOSE=true.'
    return None


def validate_model(model: str) -> str:
    if model not in MODEL_LABELS:
        raise ValueError('unknown pose model; choose lite, full, heavy or vitpose')
    reason = unavailable_reason(model)
    if reason:
        raise ModelUnavailable(reason)
    return model


def model_choices() -> list[dict]:
    return [{'id': name, 'label': label, 'available': unavailable_reason(name) is None,
             'unavailable_reason': unavailable_reason(name)}
            for name, label in MODEL_LABELS.items()]
