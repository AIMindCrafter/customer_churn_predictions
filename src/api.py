"""FastAPI inference service for churn prediction."""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import load_feature_columns, load_models
from src.validation import validate_inference_input
from src.monitoring import log_prediction
from src.logger import get_logger

logger = get_logger(__name__)

# ── Prometheus-style counters ────────────────────────────────
_metrics = {
    "prediction_requests_total": 0,
    "prediction_errors_total": 0,
    "prediction_latency_seconds_sum": 0.0,
    "prediction_churn_positive_total": 0,
    "prediction_churn_negative_total": 0,
}


# ── Request / Response schemas ──────────────────────────────

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    model_type: str = Field(default="bayesian", pattern="^(grid|bayesian)$")


class PredictResponse(BaseModel):
    model_type: str
    count: int
    predictions: List[int]
    churn_probability: List[float]


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    model_type: str = Field(default="bayesian", pattern="^(grid|bayesian)$")


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    version="2.0.0",
    description="Production-ready customer churn prediction with MLOps",
)

# Mount frontend static files (only if directory exists)
if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


def _align_features(df: pd.DataFrame, expected_columns: Optional[List[str]]) -> pd.DataFrame:
    if expected_columns is None:
        return df
    return df.reindex(columns=expected_columns, fill_value=0)


def _align_to_model_shape(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Ensure input feature count matches model expectation."""
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        return df

    current = df.shape[1]
    if current == expected:
        return df
    if current < expected:
        for idx in range(expected - current):
            df[f"__pad_{idx}"] = 0
        return df
    return df.iloc[:, :expected]


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []
    for key, value in _metrics.items():
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


@app.post("/v1/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.time()
    _metrics["prediction_requests_total"] += 1

    try:
        # Validate input
        issues = validate_inference_input(payload.records)
        if issues:
            logger.warning("Input validation issues: %s", issues)

        grid_model, bayes_model, scaler = load_models()
        feature_columns = load_feature_columns()

        input_df = pd.DataFrame(payload.records)
        if "Churn" in input_df.columns:
            input_df = input_df.drop(columns=["Churn"])

        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = _align_features(input_df, feature_columns)
        model = bayes_model if payload.model_type == "bayesian" else grid_model

        input_df = _align_to_model_shape(input_df, model)
        transformed = scaler.transform(input_df)

        pred = model.predict(transformed).tolist()
        proba = model.predict_proba(transformed)[:, 1].tolist()

        # Update metrics
        _metrics["prediction_churn_positive_total"] += sum(pred)
        _metrics["prediction_churn_negative_total"] += len(pred) - sum(pred)

        # Log prediction for monitoring
        log_prediction(payload.records, pred, proba, payload.model_type)

        response = PredictResponse(
            model_type=payload.model_type,
            count=len(pred),
            predictions=[int(v) for v in pred],
            churn_probability=[float(v) for v in proba],
        )

        elapsed = time.time() - start
        _metrics["prediction_latency_seconds_sum"] += elapsed
        logger.info("Prediction served in %.3fs (%d records, model=%s)",
                     elapsed, len(pred), payload.model_type)
        return response

    except Exception as exc:
        _metrics["prediction_errors_total"] += 1
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
