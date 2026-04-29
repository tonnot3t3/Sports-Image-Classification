# System Architecture

```mermaid
flowchart TB
    A[Client<br/>curl / Postman / JMeter] -->|POST /predict<br/>multipart| B[FastAPI<br/>async route]
    B --> V{Validation<br/>MIME, size, decode}
    V -->|400 / 413 / 415| A
    V -->|valid| P[ProcessPoolExecutor<br/>N workers]
    P --> O[ONNX Runtime<br/>vit_sports_int8.onnx<br/>89 MB INT8]
    O --> R[Top-K JSON<br/>label + score]
    R --> A

    subgraph Meta endpoints
        H[/GET /health/]
        I[/GET /info/]
        D[/GET /docs/]
    end
    A -.-> H
    A -.-> I
    A -.-> D
```

## CI/CD pipeline

```mermaid
flowchart LR
    G[git push] --> T[pytest]
    T -->|fail| F[Stop / surface error]
    T -->|pass| BD[docker build]
    BD -->|fail| F
    BD -->|pass + branch=main| H[Push to<br/>HF Spaces]
    H --> S[HF builds + serves<br/>Docker container]
```

## Request lifecycle

1. Client sends `POST /predict` with `multipart/form-data` (`file=@image.jpg`).
2. FastAPI's async route reads bytes in 64 KB chunks; aborts at the size cap.
3. MIME whitelist + non-empty check → 415 / 400 if violated.
4. Bytes are forwarded to `ProcessPoolExecutor` via
   `loop.run_in_executor(pool, predict, image_bytes, top_k)`.
5. Worker preprocesses (resize → normalize → NCHW float32) and runs the
   INT8 ONNX session.
6. Worker softmaxes the logits, picks top-K, returns `(labels, time_ms)`.
7. Route serializes `PredictResponse` (Pydantic) and returns JSON.
