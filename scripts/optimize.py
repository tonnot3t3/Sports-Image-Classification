"""Convert a fine-tuned ViT checkpoint to ONNX (FP32) and INT8 (dynamic).

Pipeline:
    1. Load the (fine-tuned) ViT model from --model_dir.
    2. Export it to ONNX using `optimum.onnxruntime`.
    3. Run dynamic quantization (INT8 weights, FP32 activations) using
       `onnxruntime.quantization.quantize_dynamic`.
    4. Benchmark all three artifacts (PyTorch FP32, ONNX FP32, ONNX INT8)
       on a synthetic batch and print/save a comparison table.

Usage:
    python scripts/optimize.py \\
        --model_dir ./vit_sports_finetuned \\
        --output_dir ./onnx_models \\
        --num_warmup 5 --num_iters 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForImageClassification
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_dir",
        default="./vit_sports_finetuned",
        help=(
            "Path to a fine-tuned model.  Falls back to "
            "google/vit-base-patch16-224 if the directory does not exist."
        ),
    )
    p.add_argument("--output_dir", default="./onnx_models")
    p.add_argument("--num_warmup", type=int, default=5)
    p.add_argument("--num_iters", type=int, default=30)
    p.add_argument(
        "--report_path",
        default="./docs/optimization_results.json",
        help="Where to write the JSON benchmark report.",
    )
    return p.parse_args()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def make_dummy_input(processor: ViTImageProcessor) -> dict:
    """Synthetic 224x224 RGB image -> ViT-ready tensors."""
    img = Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    return processor(images=img, return_tensors="pt")


# --------------------------------------------------------------------------- #
#  Benchmark helpers
# --------------------------------------------------------------------------- #

def bench_pytorch(model: torch.nn.Module, sample, n_warm: int, n_iters: int) -> dict:
    model.eval()
    pixel_values = sample["pixel_values"]
    with torch.no_grad():
        for _ in range(n_warm):
            model(pixel_values=pixel_values)
        timings = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(pixel_values=pixel_values)
            timings.append((time.perf_counter() - t0) * 1000)
    return _summarize(timings)


def bench_onnx(session, sample, n_warm: int, n_iters: int) -> dict:
    pixel_values = sample["pixel_values"].numpy().astype(np.float32)
    input_name = session.get_inputs()[0].name
    for _ in range(n_warm):
        session.run(None, {input_name: pixel_values})
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: pixel_values})
        timings.append((time.perf_counter() - t0) * 1000)
    return _summarize(timings)


def _summarize(timings):
    arr = np.array(timings, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        print(f"[warn] {model_dir} not found — falling back to google/vit-base-patch16-224 (no fine-tuning).")
        source = "google/vit-base-patch16-224"
    else:
        source = str(model_dir)

    # ----- 1) Load PyTorch model + processor --------------------------------
    print(f"[1/4] Loading PyTorch model from {source} ...")
    pt_model = ViTForImageClassification.from_pretrained(source)
    processor = ViTImageProcessor.from_pretrained(source)
    sample = make_dummy_input(processor)

    # Save the *baseline* model into out_dir so we can measure its on-disk size.
    pt_dir = out_dir / "pytorch_baseline"
    pt_model.save_pretrained(pt_dir)
    pt_size = sum(p.stat().st_size for p in pt_dir.rglob("*") if p.is_file()) / (1024 * 1024)

    # ----- 2) Export to ONNX (FP32) -----------------------------------------
    print("[2/4] Exporting to ONNX (FP32) ...")
    onnx_dir = out_dir / "onnx_fp32"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    ort_model = ORTModelForImageClassification.from_pretrained(source, export=True)
    ort_model.save_pretrained(onnx_dir)
    onnx_fp32_path = onnx_dir / "model.onnx"
    onnx_fp32_size = file_size_mb(onnx_fp32_path)

    # ----- 3) Dynamic quantization (INT8) -----------------------------------
    print("[3/4] Dynamic quantization -> INT8 ...")
    onnx_int8_path = out_dir / "vit_sports_int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
        # Per-channel quant gives noticeably better accuracy on transformer attn weights.
        per_channel=True,
        reduce_range=False,
    )
    onnx_int8_size = file_size_mb(onnx_int8_path)

    # ----- 4) Benchmark -----------------------------------------------------
    print("[4/4] Benchmarking (CPU, batch=1) ...")
    import onnxruntime as ort

    sess_fp32 = ort.InferenceSession(
        str(onnx_fp32_path), providers=["CPUExecutionProvider"]
    )
    sess_int8 = ort.InferenceSession(
        str(onnx_int8_path), providers=["CPUExecutionProvider"]
    )

    pt_stats = bench_pytorch(pt_model, sample, args.num_warmup, args.num_iters)
    fp32_stats = bench_onnx(sess_fp32, sample, args.num_warmup, args.num_iters)
    int8_stats = bench_onnx(sess_int8, sample, args.num_warmup, args.num_iters)

    # ----- Report -----------------------------------------------------------
    report = {
        "source_model": source,
        "device": "cpu",
        "batch_size": 1,
        "num_warmup": args.num_warmup,
        "num_iters": args.num_iters,
        "results": {
            "pytorch_fp32": {
                "size_mb": round(pt_size, 2),
                **{k: round(v, 2) for k, v in pt_stats.items()},
            },
            "onnx_fp32": {
                "size_mb": round(onnx_fp32_size, 2),
                **{k: round(v, 2) for k, v in fp32_stats.items()},
            },
            "onnx_int8_dynamic": {
                "size_mb": round(onnx_int8_size, 2),
                **{k: round(v, 2) for k, v in int8_stats.items()},
            },
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    # Pretty print
    print()
    print("=" * 78)
    print(f"{'Variant':<24} {'Size (MB)':>10} {'Mean (ms)':>11} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 78)
    for name, key in [
        ("PyTorch FP32 (baseline)", "pytorch_fp32"),
        ("ONNX FP32", "onnx_fp32"),
        ("ONNX INT8 (dynamic)", "onnx_int8_dynamic"),
    ]:
        r = report["results"][key]
        print(
            f"{name:<24} {r['size_mb']:>10.2f} {r['mean_ms']:>11.2f} "
            f"{r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f}"
        )
    print("=" * 78)
    print(f"Quantized model saved to: {onnx_int8_path}")
    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()
