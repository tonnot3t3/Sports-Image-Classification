---
title: Sports Image Classifier
emoji: 🏆
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: ViT-base INT8 ONNX classifier for 100 sports (FastAPI demo).
---

# Sports Image Classification API

> High-throughput Image Classification Service powered by an INT8-quantized
> **ViT-base-patch16-224** ONNX model, fine-tuned on the
> [Kaggle Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification/data?select=sports.csv) (100 classes).

This repository implements the full MLOps lifecycle from the project
assignment:

1. **Model Optimization** — convert the fine-tuned ViT model to ONNX,
   then dynamic-quantize it to INT8.
2. **API Development** — FastAPI + `ProcessPoolExecutor` for CPU-bound
   inference, Pydantic-driven validation, production-grade error
   handling.
3. **Automation & CI/CD** — `pytest` + GitHub Actions that build the
   Docker image and auto-deploy to Hugging Face Spaces.
4. **Performance Testing** — JMeter `.jmx` plan for load tests, plus a
   Postman collection.

---

## Repository layout

```
sports-vit-mlops/
├── app/                    # FastAPI application
│   ├── main.py             #   ── HTTP layer (async, error handling)
│   ├── inference.py        #   ── ONNX session + workers
│   ├── schemas.py          #   ── Pydantic request/response models
│   ├── config.py           #   ── pydantic-settings (env vars)
│   └── labels.json         #   ── 100 sport class names
├── onnx_models/            # produced by scripts/optimize.py
├── scripts/
│   ├── train.py            # fine-tune ViT on sports dataset
│   ├── optimize.py         # PyTorch -> ONNX -> INT8 + benchmark
│   └── benchmark.py        # ad-hoc latency benchmark
├── tests/                  # pytest unit tests
├── jmeter/load_test.jmx    # JMeter test plan
├── postman/collection.json # Postman collection
├── docs/                   # Project Report (PDF) and figures
├── .github/workflows/
│   └── ci-cd.yml           # Test + build + deploy to HF Spaces
├── Dockerfile
├── requirements.txt        # runtime deps (slim)
├── requirements-dev.txt    # adds torch / transformers / pytest
└── README.md
```

---

## Quickstart

### 1) Setup

```bash
git clone <your-fork>.git sports-vit-mlops
cd sports-vit-mlops
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2) Fine-tune ViT on the Sports dataset (optional but recommended)

Download the Kaggle dataset.  The simplest way is via **`kagglehub`**
(it caches under `~/.cache/kagglehub/...` and is happy on Windows / Mac
/ Linux):

```bash
pip install kagglehub
python scripts/download_data.py        # creates ./sports_dataset -> cache
```

Equivalent one-liner if you prefer doing it inline:

```python
import kagglehub
path = kagglehub.dataset_download("gpiosenka/sports-classification")
print("Path to dataset files:", path)
```

Then fine-tune:

```bash
python scripts/train.py --data_dir ./sports_dataset --output_dir ./vit_sports_finetuned --epochs 5
```

Alternative (Kaggle CLI):

```bash
kaggle datasets download -d gpiosenka/sports-classification
unzip sports-classification.zip -d sports_dataset/
```

Outputs:
- `./vit_sports_finetuned/` — fine-tuned PyTorch model + processor
- `app/labels.json` — class name list (used by the API at runtime)

### 3) Optimize: PyTorch → ONNX → INT8

```bash
python scripts/optimize.py --model_dir ./vit_sports_finetuned --output_dir ./onnx_models
```

This produces:
- `onnx_models/onnx_fp32/model.onnx` — ONNX FP32 model
- `onnx_models/vit_sports_int8.onnx` — INT8 dynamic-quantized model (the
  one the API loads at runtime)
- `docs/optimization_results.json` — comparison table

### 4) Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Browse http://localhost:7860/docs for the Swagger UI.
```

### 5) Run the API in Docker

```bash
docker build -t sports-vit-api .
docker run --rm -p 7860:7860 sports-vit-api
```

### 6) Test it

```bash
# Health
curl http://localhost:7860/health

# Predict (replace tennis.jpg with any sports image)
curl -X POST -F "file=@tennis.jpg" http://localhost:7860/predict
```

#### cURL: calling the API on Hugging Face Spaces

```bash
curl -X POST \
  -F "file=@tennis.jpg" \
  https://<your-username>-sports-vit-api.hf.space/predict
```

A successful response looks like:

```json
{
  "filename": "tennis.jpg",
  "model": "vit_sports_int8.onnx",
  "inference_ms": 81.4,
  "predictions": [
    {"label": "tennis",    "score": 0.9231},
    {"label": "badminton", "score": 0.0413},
    {"label": "table tennis", "score": 0.0118}
  ]
}
```

---

## API

| Method | Path       | Description                               |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/`        | Service banner                            |
| GET    | `/health`  | Liveness/readiness probe                  |
| GET    | `/info`    | Model metadata + worker count             |
| POST   | `/predict` | Classify an uploaded image (`multipart/form-data`, field `file`) |
| GET    | `/docs`    | Swagger UI                                |

### Error handling matrix

| Situation                                | HTTP status | `detail`                                |
| ---------------------------------------- | ----------- | --------------------------------------- |
| Empty body / no file                     | 400         | `Uploaded file is empty.`               |
| Unsupported MIME (`text/plain`, etc.)    | 415         | `Content-Type '...' is not supported.`  |
| Image > 5 MB                             | 413         | `Image is too large: ... bytes.`        |
| Corrupt / non-decodable image bytes      | 400         | `Could not decode image: ...`           |
| Worker pool not yet ready                | 503         | `Worker pool is not ready yet ...`      |
| Anything else                            | 500         | `Internal server error: <ExceptionType>`|

---

## Generating the Project Report PDF

```bash
pip install -r requirements-dev.txt   # ensures reportlab + matplotlib
python scripts/optimize.py            # populates docs/optimization_results.json
python scripts/generate_report.py     # writes docs/Project_Report.pdf
```

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push and PR:

1. **lint + test** — `pytest` across the `tests/` suite.
2. **build** — builds the Docker image to validate `Dockerfile`.
3. **deploy** *(only on `main`, only if tests pass 100%)* — pushes the
   repository to your Hugging Face Space using `HF_TOKEN`.

Set these GitHub repository secrets:
- `HF_TOKEN` — HF write token (User Settings → Access Tokens).
- `HF_USERNAME` — your HF username.
- `HF_SPACE_NAME` — name of the Space (e.g. `sports-vit-api`).

Create the matching Space ahead of time (Docker SDK).

> **Note on the model file.** The INT8 ONNX model (~ 90 MB) is tracked
> through **Git LFS** (see `.gitattributes`).  Run `git lfs install`
> once on your machine and on the CI runner, then commit the file
> normally.  Hugging Face Spaces understands LFS pointers natively.

---

## Testing artifacts

- `jmeter/load_test.jmx` — load test plan.  Run with:
  ```bash
  jmeter -n -t jmeter/load_test.jmx -l jmeter/results/result.jtl \
         -e -o jmeter/report \
         -Jhost=localhost -Jport=7860 -Jthreads=50 -Jduration=120
  ```
- `postman/collection.json` — import into Postman to hit `/health`,
  `/info`, `/predict`.
- `docs/Project_Report.pdf` — full project write-up with optimization
  results, architecture diagram, and JMeter analysis.

---

## License

Built for an MLOps coursework assignment.  Model weights inherit the
license of `google/vit-base-patch16-224` (Apache-2.0).
