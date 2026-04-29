"""Standalone benchmark of the deployed INT8 ONNX model.

Useful for quick post-deployment sanity checks (e.g. on the HF Spaces
container) without re-running the whole optimization pipeline.

Usage:
    python scripts/benchmark.py --model_path onnx_models/vit_sports_int8.onnx \\
        --image tests/fixtures/tennis.jpg --iters 50
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    return (arr - mean) / std


def main() -> None:
    args = parse_args()
    sess = ort.InferenceSession(
        args.model_path, providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0].name
    tensor = preprocess(Path(args.image)).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {inp: tensor})

    timings = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {inp: tensor})
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    print(f"iters={args.iters}  mean={arr.mean():.2f}ms  "
          f"p50={np.median(arr):.2f}ms  p95={np.percentile(arr, 95):.2f}ms  "
          f"p99={np.percentile(arr, 99):.2f}ms")


if __name__ == "__main__":
    main()
