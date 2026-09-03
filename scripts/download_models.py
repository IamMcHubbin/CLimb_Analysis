#!/usr/bin/env python3
"""Fetch pose model files into the model directory.

Run at image build time so the container does not download anything on first
request.

Usage:
    python scripts/download_models.py            # the configured variant
    python scripts/download_models.py lite full  # named variants
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.pose.model_zoo import MODEL_URLS, ensure_model  # noqa: E402


def main(argv: list[str]) -> int:
    variants = argv or [settings.pose_model]
    unknown = [variant for variant in variants if variant not in MODEL_URLS]
    if unknown:
        print(f"unknown model variant(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"expected one of: {', '.join(sorted(MODEL_URLS))}", file=sys.stderr)
        return 2

    settings.ensure_dirs()
    for variant in variants:
        path = ensure_model(variant)
        print(f"{variant}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
