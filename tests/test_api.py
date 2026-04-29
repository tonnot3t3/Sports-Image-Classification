"""End-to-end HTTP tests.

Each test in this module uses ``app_client`` which itself depends on
``model_available``.  When the ONNX model is missing the whole class is
skipped so CI on a clean checkout still passes.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image


def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["workers"] >= 1


def test_info(app_client):
    r = app_client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["num_labels"] == 100
    assert body["max_image_bytes"] > 0


def test_predict_happy_path(app_client, sample_jpeg_bytes):
    r = app_client.post(
        "/predict",
        files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["filename"] == "img.jpg"
    assert "model" in body
    assert body["inference_ms"] > 0

    preds = body["predictions"]
    assert len(preds) == 5
    # Top-k must be sorted descending.
    assert all(preds[i]["score"] >= preds[i + 1]["score"] for i in range(len(preds) - 1))
    # Probabilities are valid.
    for p in preds:
        assert 0.0 <= p["score"] <= 1.0
        assert isinstance(p["label"], str)


def test_predict_top_label_is_in_labels_file(app_client, sample_jpeg_bytes):
    """The top-1 label must exist in our labels.json (sanity / model
    integrity check)."""
    import json
    from pathlib import Path

    labels = set(json.loads(
        (Path(__file__).resolve().parent.parent / "app" / "labels.json").read_text()
    ))
    r = app_client.post(
        "/predict",
        files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    body = r.json()
    assert body["predictions"][0]["label"] in labels


def test_predict_rejects_non_image_mime(app_client):
    r = app_client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 415
    assert "not supported" in r.json()["detail"].lower()


def test_predict_rejects_corrupt_image(app_client, corrupt_image_bytes):
    r = app_client.post(
        "/predict",
        files={"file": ("bad.png", corrupt_image_bytes, "image/png")},
    )
    assert r.status_code == 400
    assert "decode" in r.json()["detail"].lower()


def test_predict_rejects_empty_file(app_client):
    r = app_client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_rejects_oversized_file(app_client):
    """6 MB image should be rejected (limit is 5 MB by default)."""
    big_payload = b"\xff\xd8\xff\xe0" + b"\x00" * (6 * 1024 * 1024)
    r = app_client.post(
        "/predict",
        files={"file": ("huge.jpg", big_payload, "image/jpeg")},
    )
    assert r.status_code == 413


@pytest.mark.parametrize("ext,mime,fmt", [
    ("jpg", "image/jpeg", "JPEG"),
    ("png", "image/png", "PNG"),
    ("webp", "image/webp", "WEBP"),
])
def test_predict_supports_multiple_image_formats(app_client, ext, mime, fmt):
    img = Image.new("RGB", (224, 224), color=(0, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    r = app_client.post(
        "/predict",
        files={"file": (f"x.{ext}", buf.getvalue(), mime)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["score"] > 0
