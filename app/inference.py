"""ONNX-based ViT inference logic.

The heavy CPU work (matrix multiplications inside the transformer
blocks) is run inside worker processes via `concurrent.futures.
ProcessPoolExecutor` so the FastAPI event loop is never blocked.

Each worker loads its own copy of the ONNX session at import time
(see `_init_worker`) so we pay the load cost exactly once per worker.

Pre-processing intentionally re-implements ViT's image transform in
NumPy/Pillow only.  This keeps the runtime image free of PyTorch and
shaves ~700 MB off the final Docker layer.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError


# ViT-base-patch16-224 normalization (matches the HF feature extractor).
_IMAGE_SIZE = 224
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)


# These globals live *inside each worker process* after _init_worker runs.
_SESSION: ort.InferenceSession | None = None
_INPUT_NAME: str | None = None
_LABELS: List[str] = []


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def load_labels(labels_path: Path) -> List[str]:
    """Load class labels from a JSON file.

    The file is expected to be a JSON list whose i-th entry is the label
    for class index i.
    """
    with open(labels_path, "r", encoding="utf-8") as fp:
        labels = json.load(fp)
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"{labels_path} must be a JSON list of strings.")
    return labels


def _init_worker(model_path: str, labels_path: str) -> None:
    """ProcessPoolExecutor `initializer` — runs once per worker."""
    global _SESSION, _INPUT_NAME, _LABELS

    sess_options = ort.SessionOptions()
    # Single-thread per worker; the *pool* gives us parallelism.
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    _SESSION = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _INPUT_NAME = _SESSION.get_inputs()[0].name
    _LABELS = load_labels(Path(labels_path))


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode + resize + normalize -> float32 NCHW tensor.

    Raises:
        ValueError: when the bytes are not a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, 0..1
    arr = arr.transpose(2, 0, 1)[None, ...]           # NCHW
    arr = (arr - _MEAN) / _STD
    return arr.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def predict(image_bytes: bytes, top_k: int = 5) -> Tuple[List[Tuple[str, float]], float]:
    """Run inference inside the worker process.

    Returns:
        (top_k predictions, inference_time_in_ms)
    """
    if _SESSION is None or _INPUT_NAME is None:
        raise RuntimeError(
            "Inference session is not initialised. "
            "Did you forget to pass `_init_worker` to ProcessPoolExecutor?"
        )

    t0 = time.perf_counter()
    tensor = _preprocess(image_bytes)
    logits = _SESSION.run(None, {_INPUT_NAME: tensor})[0]  # (1, num_labels)
    probs = _softmax(logits)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    k = min(top_k, probs.shape[0])
    idx = np.argpartition(-probs, kth=k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]

    results = [(_LABELS[i] if i < len(_LABELS) else str(i), float(probs[i])) for i in idx]
    return results, elapsed_ms


# --------------------------------------------------------------------------- #
#  Helper used by the parent process for /info and tests
# --------------------------------------------------------------------------- #

def model_info(model_path: Path) -> dict:
    """Cheap metadata used in /info — does NOT load the session."""
    return {
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size if model_path.exists() else 0,
    }
