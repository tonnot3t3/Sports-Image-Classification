"""Pytest fixtures.

The test suite covers two layers:

1. *Pure* preprocessing logic (no model required) — runs in CI on every
   push.
2. *End-to-end* HTTP behaviour with the real ONNX model — only runs if
   ``onnx_models/vit_sports_int8.onnx`` is present on the runner.  Tests
   in this group skip cleanly when the model is missing so CI does not
   need to download / regenerate it.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "onnx_models" / "vit_sports_int8.onnx"
LABELS_PATH = ROOT / "app" / "labels.json"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def model_available() -> bool:
    """True only if the *real* ONNX model is on disk.

    On CI (without `lfs: true`) the file may exist as a small Git LFS
    pointer (~ 130 bytes).  We require at least 1 MB to call it real.
    """
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        return False
    return MODEL_PATH.stat().st_size > 1_000_000


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """A 224x224 solid-blue JPEG."""
    img = Image.new("RGB", (224, 224), color=(40, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(255, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Bytes that look like an image MIME but are not valid image data."""
    return b"\x89PNG\r\n\x1a\n" + b"this-is-not-a-real-png-payload" * 10


@pytest.fixture
def app_client(model_available, monkeypatch):
    """FastAPI TestClient.

    Skips the test if the ONNX model is not available.
    """
    if not model_available:
        pytest.skip(
            "ONNX model not present (run scripts/optimize.py first). "
            "End-to-end test skipped."
        )

    # Make sure the app uses the in-repo paths during tests.
    monkeypatch.setenv("MODEL_PATH", str(MODEL_PATH))
    monkeypatch.setenv("LABELS_PATH", str(LABELS_PATH))

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client
