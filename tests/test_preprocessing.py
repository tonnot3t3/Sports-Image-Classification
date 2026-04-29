"""Pure preprocessing tests — run without the ONNX model present."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.inference import _preprocess


def test_preprocess_shape_and_dtype(sample_jpeg_bytes):
    arr = _preprocess(sample_jpeg_bytes)
    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32


def test_preprocess_normalisation_range(sample_jpeg_bytes):
    """Mean=0.5, Std=0.5 maps [0,1] -> [-1,1]."""
    arr = _preprocess(sample_jpeg_bytes)
    assert arr.min() >= -1.0001
    assert arr.max() <= 1.0001


def test_preprocess_handles_grayscale():
    img = Image.new("L", (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    arr = _preprocess(buf.getvalue())
    assert arr.shape == (1, 3, 224, 224)  # promoted to RGB


def test_preprocess_rejects_corrupt_bytes(corrupt_image_bytes):
    with pytest.raises(ValueError):
        _preprocess(corrupt_image_bytes)


def test_labels_file_is_well_formed(project_root: Path):
    labels = json.loads((project_root / "app" / "labels.json").read_text())
    assert isinstance(labels, list)
    assert all(isinstance(x, str) for x in labels)
    assert len(labels) == 100, f"expected 100 sports labels, got {len(labels)}"
    assert len(set(labels)) == len(labels), "labels must be unique"
