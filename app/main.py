"""FastAPI entry-point.

The HTTP layer is intentionally thin: it validates the incoming
multipart upload, then ships the bytes to a worker process via
`ProcessPoolExecutor`.  The event loop never executes a single
matrix multiply, so the API stays responsive under load.
"""
from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import __version__
from .config import settings
from .inference import _init_worker, load_labels, model_info, predict
from .schemas import (
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    PredictResponse,
    Prediction,
)


# --------------------------------------------------------------------------- #
#  Lifespan: spin up / tear down the ProcessPoolExecutor
# --------------------------------------------------------------------------- #

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the worker pool when the server boots and shut it down cleanly."""
    if not settings.model_path.exists():
        raise RuntimeError(
            f"Model file not found at {settings.model_path}. "
            f"Run `python scripts/optimize.py` first."
        )
    if not settings.labels_path.exists():
        raise RuntimeError(
            f"Labels file not found at {settings.labels_path}."
        )

    labels = load_labels(settings.labels_path)
    logger.info(
        f"Spinning up {settings.worker_processes} workers "
        f"(model={settings.model_path.name}, num_labels={len(labels)})"
    )

    pool = ProcessPoolExecutor(
        max_workers=settings.worker_processes,
        initializer=_init_worker,
        initargs=(str(settings.model_path), str(settings.labels_path)),
    )
    # Eagerly warm up workers so the first user request doesn't pay the cold-start cost.
    try:
        # Trigger initializer execution by submitting a no-op-ish dummy task.
        # We use a 1x1 PNG as the warm-up image.
        warm_up_bytes = _DUMMY_PNG
        loop = asyncio.get_running_loop()
        warm = [
            loop.run_in_executor(pool, predict, warm_up_bytes, 1)
            for _ in range(settings.worker_processes)
        ]
        await asyncio.gather(*warm, return_exceptions=True)
        logger.info("Worker pool warm-up complete.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Warm-up failed (non-fatal): {exc}")

    app.state.pool = pool
    app.state.labels = labels
    try:
        yield
    finally:
        logger.info("Shutting down worker pool ...")
        pool.shutdown(wait=False, cancel_futures=True)


# Tiny 1x1 PNG used for the warm-up call (not committed as a file).
_DUMMY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cf00000003000100ff5be1aa0000000049454e44ae426082"
)


# --------------------------------------------------------------------------- #
#  App + middleware
# --------------------------------------------------------------------------- #

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "High-throughput Image Classification Service for sports images, "
        "powered by an INT8-quantized ViT-base-patch16-224 ONNX model."
    ),
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)

# Allow JMeter / Postman / browser callers from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Static frontend (the demo UI lives in app/static/)
# --------------------------------------------------------------------------- #

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
#  Error handlers — surface clean JSON instead of HTML stack traces
# --------------------------------------------------------------------------- #

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# --------------------------------------------------------------------------- #
#  Helper: validate the multipart upload BEFORE crossing the process boundary
# --------------------------------------------------------------------------- #

async def _read_and_validate(file: UploadFile) -> bytes:
    """Apply all input checks.  Raises HTTPException on any failure."""
    # 1) MIME type
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Content-Type '{file.content_type}' is not supported. "
                f"Allowed: {', '.join(settings.allowed_content_types)}."
            ),
        )

    # 2) Size cap (read incrementally so a 1 GB upload doesn't OOM us)
    chunks: list[bytes] = []
    total = 0
    chunk_size = 64 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Image is too large: {total} > "
                    f"{settings.max_image_bytes} bytes."
                ),
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return b"".join(chunks)


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False, response_model=None)
async def root():
    """Serve the demo web UI when present, fall back to a JSON banner."""
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html), media_type="text/html")
    return JSONResponse({
        "service": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
    })


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    pool = getattr(request.app.state, "pool", None)
    return HealthResponse(
        status="ok",
        model_loaded=pool is not None,
        workers=settings.worker_processes,
    )


@app.get("/info", response_model=InfoResponse, tags=["meta"])
async def info(request: Request) -> InfoResponse:
    labels = getattr(request.app.state, "labels", [])
    info_meta = model_info(settings.model_path)
    return InfoResponse(
        api=settings.api_title,
        version=__version__,
        model_path=info_meta["model_path"],
        num_labels=len(labels),
        workers=settings.worker_processes,
        max_image_bytes=settings.max_image_bytes,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request / corrupt image"},
        413: {"model": ErrorResponse, "description": "Image too large"},
        415: {"model": ErrorResponse, "description": "Unsupported MIME type"},
        503: {"model": ErrorResponse, "description": "Worker pool not ready"},
    },
)
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Sports image to classify."),
) -> PredictResponse:
    """Classify an uploaded sports image.

    The endpoint is `async def` so it never blocks the event loop.  The
    actual ONNX session call happens inside a worker process via
    `loop.run_in_executor(...)`.
    """
    pool: ProcessPoolExecutor | None = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker pool is not ready yet — try again in a moment.",
        )

    image_bytes = await _read_and_validate(file)

    loop = asyncio.get_running_loop()
    try:
        results, elapsed_ms = await loop.run_in_executor(
            pool, predict, image_bytes, settings.top_k,
        )
    except ValueError as exc:
        # `predict` re-raises ValueError on undecodable / corrupt images.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}",
        ) from exc

    return PredictResponse(
        filename=file.filename or "uploaded.bin",
        model=Path(settings.model_path).name,
        inference_ms=round(elapsed_ms, 2),
        predictions=[Prediction(label=lbl, score=score) for lbl, score in results],
    )


# Allow `python -m app.main` for ad-hoc local runs.
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level=settings.log_level.lower(),
    )
