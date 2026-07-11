"""FastAPI service exposing the heart disease risk model as /predict.

Also includes basic request logging and a lightweight /metrics endpoint
for monitoring (Task 8).
"""
import logging
import pathlib
import time
from collections import Counter

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from api.schemas import PatientFeatures, PredictionResponse
from src.pipeline import ALL_FEATURES

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
LOG_PATH = ROOT / "logs" / "api.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

_model = None
_metrics = Counter()


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    _metrics["requests_total"] += 1
    _metrics[f"status_{response.status_code}"] += 1

    logger.info(
        "%s %s -> %s (%.2fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return dict(_metrics)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in sorted(_metrics.items())
    )
    return f"""
    <html>
      <head><title>Heart Disease API - Monitoring</title></head>
      <body style="font-family: sans-serif; max-width: 480px; margin: 40px auto;">
        <h2>API Monitoring Dashboard</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="width: 100%; border-collapse: collapse;">
          <tr><th>Metric</th><th>Value</th></tr>
          {rows or '<tr><td colspan="2">No requests recorded yet</td></tr>'}
        </table>
        <p><a href="/docs">Swagger UI</a> | <a href="/metrics">Raw metrics JSON</a></p>
      </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    model = get_model()
    row = pd.DataFrame([patient.model_dump()])[ALL_FEATURES]

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0][pred]

    _metrics["predictions_total"] += 1
    _metrics[f"predictions_class_{pred}"] += 1

    return PredictionResponse(
        prediction=pred,
        label="Disease" if pred == 1 else "No Disease",
        confidence=round(float(proba), 4),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
