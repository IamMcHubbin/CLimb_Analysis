"""Writing an incoming upload to disk without holding it in memory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from app.ingest.errors import IngestError


class UploadTooLarge(IngestError):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds the {limit_bytes} byte limit")
        self.limit_bytes = limit_bytes


class EmptyUpload(IngestError):
    def __init__(self) -> None:
        super().__init__("upload is empty")


async def save_stream(
    chunks: AsyncIterator[bytes],
    destination: Path,
    max_bytes: int,
) -> int:
    """Stream ``chunks`` to ``destination``, enforcing the size limit as we go.

    The limit is checked during the write rather than afterwards so an
    oversized upload cannot fill the disk first. A partial file is removed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLarge(max_bytes)
                handle.write(chunk)
        if written == 0:
            raise EmptyUpload()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return written
