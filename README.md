<<<<<<< HEAD
# cats_dogs_mlops
=======
# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification (Cats vs Dogs) built as a complete MLOps assignment
covering model development, packaging, CI/CD, deployment, and monitoring.

---

## Project Structure

```
cats_dogs_mlops/
├── src/
│   ├── model.py          # SimpleCNN architecture + load_model()
│   └── preprocess.py     # Transforms, image preprocessing, dataset split
├── api/
│   ├── main.py           # FastAPI app (health, predict, metrics endpoints)
│   ├── predictor.py      # Singleton inference wrapper
│   └── schemas.py        # Pydantic request/response schemas
├── tests/
│   ├── test_preprocess.py  # Unit tests — data preprocessing functions
│   └── test_predictor.py   # Unit tests — predictor + FastAPI endpoints
├── .github/workflows/
│   ├── ci.yml            # CI: test → build → push Docker image
│   └── cd.yml            # CD: pull image → deploy → smoke test
├── k8s/
│   ├── deployment.yaml   # Kubernetes Deployment (2 replicas, rolling update)
│   └── service.yaml      # Kubernetes NodePort Service
├── monitoring/
│   └── prometheus.yml    # Prometheus scrape config
├── train.py              # End-to-end training script with MLflow tracking
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Compose: API + MLflow server + Prometheus
├── dvc.yaml              # DVC pipeline stages (preprocess → train)
├── params.yaml           # Hyperparameters consumed by DVC + train.py
├── requirements.txt      # Pinned runtime dependencies
├── requirements-dev.txt  # + pytest / httpx for testing
└── smoke_test.sh         # Post-deploy smoke test script
```

---

## Milestones

### M1 — Model Development & Experiment Tracking

**Dataset setup (Kaggle Cats vs Dogs)**

```bash
# 1. Download dataset from Kaggle
kaggle datasets download -d salader/dogs-vs-cats
unzip dogs-vs-cats.zip -d data/raw

# 2. Version data with DVC
git init
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "feat: add raw dataset via DVC"

# 3. Run preprocessing + training via DVC pipeline
dvc repro          # runs preprocess → train stages defined in dvc.yaml
```

**Manual training (no DVC)**

```bash
pip install -r requirements.txt

python train.py \
  --data_dir   data/processed \
  --model_path models/best_model.pt \
  --epochs     15 \
  --batch_size 32 \
  --lr         0.001
```

MLflow logs parameters, per-epoch metrics, loss curves, confusion matrix, and
registers the model under `cats_dogs_classifier`.

**View MLflow UI**

```bash
mlflow ui --backend-store-uri mlruns --port 5000
# open http://localhost:5000
```

---

### M2 — Model Packaging & Containerization

**Run the API locally (no Docker)**

```bash
MODEL_PATH=models/best_model.pt uvicorn api.main:app --reload --port 8000
```

**Health check**

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"version":"1.0.0"}
```

**Prediction via curl**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/your/cat.jpg;type=image/jpeg"
# {"predicted_class":"Cat","confidence":0.9231,"probabilities":{"Cat":0.9231,"Dog":0.0769}}
```

**Build and run with Docker**

```bash
docker build -t cats-dogs-classifier:latest .

docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  cats-dogs-classifier:latest
```

**Run full stack with Docker Compose**

```bash
docker compose up -d
# API  → http://localhost:8000
# MLflow → http://localhost:5000
# Prometheus → http://localhost:9090
```

---

### M3 — CI Pipeline

The CI workflow (`.github/workflows/ci.yml`) triggers on every push / PR:

1. Checks out the repo
2. Installs `requirements-dev.txt`
3. Runs `pytest tests/ --cov` (must pass before Docker build)
4. Builds the Docker image (multi-stage)
5. Pushes tagged image to Docker Hub

**Required GitHub Secrets**

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

**Run tests locally**

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src --cov=api
```

---

### M4 — CD Pipeline & Deployment

The CD workflow (`.github/workflows/cd.yml`) triggers on pushes to `main`:

1. Pulls the latest image from Docker Hub
2. Starts the service with `docker compose up -d api`
3. Waits for it to become healthy
4. Runs smoke tests (health + prediction)
5. Tears down if smoke tests fail (pipeline fails)

**Kubernetes deployment (kind / minikube)**

```bash
# Start local cluster
kind create cluster --name mlops

# Apply manifests (update image name in deployment.yaml first)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check rollout
kubectl rollout status deployment/cats-dogs-api

# Access (minikube)
minikube service cats-dogs-api-svc --url
```

**Manual smoke test**

```bash
chmod +x smoke_test.sh
bash smoke_test.sh http://localhost:8000
```

---

### M5 — Monitoring & Logging

- **Request logging** — every request is logged with method, path, status code,
  and latency via FastAPI middleware (no image payloads logged).
- **Prometheus metrics** exposed at `GET /metrics`:
  - `inference_requests_total{endpoint, status}` — request counter
  - `inference_request_latency_seconds{endpoint}` — latency histogram
- **Prometheus** configured in `monitoring/prometheus.yml` and included in
  `docker-compose.yml`.

**Query examples (Prometheus UI at :9090)**

```promql
# Request rate
rate(inference_requests_total[1m])

# 95th-percentile latency
histogram_quantile(0.95, rate(inference_request_latency_seconds_bucket[5m]))
```

---

## Key Dependencies

| Library | Purpose |
|---|---|
| PyTorch 2.1 | Model training & inference |
| torchvision 0.16 | Dataset loading, transforms |
| FastAPI 0.104 | REST API |
| MLflow 2.8 | Experiment tracking |
| DVC 3.30 | Data & pipeline versioning |
| prometheus-client | Metrics exposition |
| pytest 7.4 | Unit testing |

---

## Architecture Diagram

```
[Kaggle Dataset]
      │  dvc add / dvc repro
      ▼
[data/raw] ──preprocess──► [data/processed]
                                  │
                              train.py
                                  │
                         MLflow tracks ──► [mlruns/]
                                  │
                          [models/best_model.pt]
                                  │
                         ┌────────▼────────┐
                         │  FastAPI  :8000  │
                         │  /health         │
                         │  /predict        │
                         │  /metrics        │
                         └────────┬────────┘
                                  │ Docker image
                         CI (GitHub Actions)
                          test → build → push
                                  │
                         CD (GitHub Actions)
                          pull → deploy → smoke test
                                  │
                    ┌─────────────▼─────────────┐
                    │  Docker Compose / k8s       │
                    │  + Prometheus :9090         │
                    └────────────────────────────┘
```
>>>>>>> 06aa4e6 (Updated training script)
