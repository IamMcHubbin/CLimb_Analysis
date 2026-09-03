"""Streaming an upload to disk under a size limit."""

from __future__ import annotations

import asyncio

import pytest

from app.ingest.upload import EmptyUpload, UploadTooLarge, save_stream


async def _chunks(*payloads: bytes):
    for payload in payloads:
        yield payload


def test_writes_all_chunks(tmp_path):
    destination = tmp_path / "out.bin"
    written = asyncio.run(save_stream(_chunks(b"abc", b"def"), destination, max_bytes=100))
    assert written == 6
    assert destination.read_bytes() == b"abcdef"


def test_rejects_and_removes_oversized_upload(tmp_path):
    destination = tmp_path / "out.bin"
    with pytest.raises(UploadTooLarge):
        asyncio.run(save_stream(_chunks(b"a" * 10, b"b" * 10), destination, max_bytes=15))
    # A partial file must not be left behind for ffmpeg to trip over.
    assert not destination.exists()


def test_rejects_empty_upload(tmp_path):
    destination = tmp_path / "out.bin"
    with pytest.raises(EmptyUpload):
        asyncio.run(save_stream(_chunks(), destination, max_bytes=100))
    assert not destination.exists()
