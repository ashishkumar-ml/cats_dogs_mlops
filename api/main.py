"""
main.py — FastAPI inference service for Cats vs Dogs classifier.

Endpoints
---------
GET  /health          — liveness + readiness check
POST /predict         — accepts a JPEG/PNG image, returns class + probabilities
GET  /metrics         — Prometheus text metrics (request count, latency)
"""
import logging
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)

from api.predictor import predictor
from api.schemas import HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of prediction requests in seconds",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification inference service",
    version="1.0.0",
)

APP_VERSION = "1.0.0"


@app.on_event("startup")
async def startup():
    predictor.load()
    log.info("Predictor ready. Model loaded: %s", predictor.is_loaded)


# ---------------------------------------------------------------------------
# Middleware — request / response logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start   = time.time()
    method  = request.method
    path    = request.url.path
    response = await call_next(request)
    latency  = time.time() - start
    # Do NOT log request bodies — may contain PII / image data
    log.info("%s %s -> %d (%.3fs)", method, path, response.status_code, latency)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        REQUEST_COUNT.labels(endpoint="/predict", status="4xx").inc()
        raise HTTPException(status_code=400,
                            detail="Only JPEG and PNG images are accepted.")

    start = time.time()
    try:
        image_bytes = await file.read()
        predicted_class, confidence, probabilities = predictor.predict(image_bytes)
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="5xx").inc()
        log.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.")
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    REQUEST_COUNT.labels(endpoint="/predict", status="2xx").inc()
    log.info(
        "Prediction: class=%s confidence=%.4f file=%s",
        predicted_class, confidence, file.filename,
    )
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        probabilities=probabilities,
    )


@app.get("/metrics", tags=["ops"])
def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
