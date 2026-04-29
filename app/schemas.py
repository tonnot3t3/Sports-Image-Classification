"""Pydantic schemas used by the FastAPI layer.

Pydantic v2 enforces type safety on the response side.  Input
validation for the multipart upload is done in `app/main.py` (because
the file payload itself is not a JSON-modeled object) but the same
validation philosophy applies: refuse anything malformed *before* a
worker process is touched.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, ConfigDict


class Prediction(BaseModel):
    """A single (label, score) pair."""

    model_config = ConfigDict(json_schema_extra={
        "example": {"label": "tennis", "score": 0.9231}
    })

    label: str = Field(..., description="Sport class label.")
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Softmax probability for this class (0..1).",
    )


class PredictResponse(BaseModel):
    """Response body returned by /predict."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "filename": "match.jpg",
            "model": "vit-base-patch16-224 (INT8 ONNX)",
            "inference_ms": 81.4,
            "predictions": [
                {"label": "tennis", "score": 0.9231},
                {"label": "badminton", "score": 0.0413},
            ],
        }
    })

    filename: str
    model: str
    inference_ms: float = Field(..., description="Server-side inference time (ms).")
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    # Allow `model_*` field names without pydantic warnings.
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool
    workers: int


class InfoResponse(BaseModel):
    # Allow `model_*` field names without pydantic warnings.
    model_config = ConfigDict(protected_namespaces=())

    api: str
    version: str
    model_path: str
    num_labels: int
    workers: int
    max_image_bytes: int


class ErrorResponse(BaseModel):
    """Standard error envelope used by all 4xx/5xx responses."""

    model_config = ConfigDict(json_schema_extra={
        "example": {"detail": "Uploaded file is not a valid image."}
    })

    detail: str
