"""Application configuration via environment variables.

All settings can be overridden through env vars (e.g. when running in
Docker / Hugging Face Spaces).  Pydantic-Settings does the parsing and
type validation for us.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow `model_*` field names without pydantic warnings.
        protected_namespaces=(),
    )

    # --- Model -----------------------------------------------------------
    model_path: Path = Field(
        default=PROJECT_ROOT / "onnx_models" / "vit_sports_int8.onnx",
        description="Path to the ONNX model used for inference.",
    )
    labels_path: Path = Field(
        default=PROJECT_ROOT / "app" / "labels.json",
        description="Path to the JSON file mapping class index -> label.",
    )

    # --- API limits ------------------------------------------------------
    max_image_bytes: int = Field(
        default=5 * 1024 * 1024,  # 5 MB
        description="Maximum size (in bytes) of an uploaded image.",
    )
    allowed_content_types: tuple[str, ...] = Field(
        default=("image/jpeg", "image/png", "image/webp", "image/bmp"),
        description="Accepted MIME types for /predict.",
    )

    # --- Concurrency -----------------------------------------------------
    worker_processes: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of worker processes in the inference pool.",
    )

    # --- Inference -------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)

    # --- Misc ------------------------------------------------------------
    log_level: str = Field(default="INFO")
    api_title: str = "Sports Image Classification API"
    api_version: str = "1.0.0"


settings = Settings()
