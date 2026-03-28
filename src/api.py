"""FastAPI inference service for churn prediction — Production v3.0"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import load_feature_columns, load_models
from src.validation import validate_inference_input
from src.monitoring import log_prediction
from src.logger import get_logger
from src.config import MODELS_DIR, PLOTS_DIR

logger = get_logger(__name__)

# ── Prometheus-style counters ────────────────────────────────
_metrics = {
    "prediction_requests_total": 0,
    "prediction_errors_total": 0,
    "prediction_latency_seconds_sum": 0.0,
    "prediction_churn_positive_total": 0,
    "prediction_churn_negative_total": 0,
    "uptime_start": time.time(),
}

# ── Model cache (loaded once at startup) ─────────────────────
_model_cache: dict = {}


def _load_models_into_cache():
    """Load models once and cache in memory."""
    try:
        grid_model, bayes_model, scaler = load_models()
        feature_columns = load_feature_columns()
        _model_cache["grid"] = grid_model
        _model_cache["bayesian"] = bayes_model
        _model_cache["scaler"] = scaler
        _model_cache["feature_columns"] = feature_columns
        _model_cache["loaded"] = True
        logger.info("✅ Models loaded and cached successfully")
    except FileNotFoundError as exc:
        _model_cache["loaded"] = False
        logger.warning("⚠️ Models not found: %s — run training first", exc)
    except Exception as exc:
        _model_cache["loaded"] = False
        logger.error("❌ Failed to load models: %s", exc, exc_info=True)


# ── Request / Response schemas ──────────────────────────────

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    model_type: str = Field(default="bayesian", pattern="^(grid|bayesian)$")


class PredictResponse(BaseModel):
    model_type: str
    count: int
    predictions: List[int]
    churn_probability: List[float]
    risk_levels: List[str]
    recommendations: List[List[str]]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    models_loaded: bool
    uptime_seconds: float


class ModelMetricsResponse(BaseModel):
    grid_model: Dict[str, Any]
    bayesian_model: Dict[str, Any]


# ── Lifespan (startup/shutdown) ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    _load_models_into_cache()
    yield
    logger.info("Shutting down API server")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    version="3.0.0",
    description="Production-ready customer churn prediction with MLOps",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if not os.path.isdir(_frontend_dir):
    _frontend_dir = "frontend"
if os.path.isdir(_frontend_dir):
    app.mount("/frontend", StaticFiles(directory=_frontend_dir), name="frontend")


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


def _get_risk_level(proba: float) -> str:
    """Classify probability into risk level."""
    if proba > 0.7:
        return "High"
    elif proba > 0.4:
        return "Medium"
    return "Low"


def _get_recommendations(record: dict, proba: float, pred: int) -> List[str]:
    """Generate retention recommendations based on customer profile and risk."""
    recs = []
    if pred == 0:
        return ["✅ Customer appears stable — continue standard engagement"]

    # Contract-based
    contract = record.get("Contract", "")
    if contract == "Month-to-month":
        recs.append("🔒 Offer discounted annual contract to increase commitment")

    # Tenure-based
    tenure = record.get("tenure", 0)
    if isinstance(tenure, (int, float)) and tenure < 12:
        recs.append("🎁 Provide loyalty rewards for new customers (< 1 year)")

    # Charges-based
    monthly = record.get("MonthlyCharges", 0)
    if isinstance(monthly, (int, float)) and monthly > 70:
        recs.append("💰 Review pricing — monthly charges above $70 correlate with higher churn")

    # Service-based
    if record.get("TechSupport") == "No" and record.get("InternetService") != "No":
        recs.append("🛠️ Bundle free tech support to improve service satisfaction")

    if record.get("OnlineSecurity") == "No" and record.get("InternetService") != "No":
        recs.append("🔐 Offer complementary online security for 3 months")

    if record.get("OnlineBackup") == "No" and record.get("InternetService") != "No":
        recs.append("☁️ Add online backup service at no extra cost for retention")

    # Payment method
    if record.get("PaymentMethod") == "Electronic check":
        recs.append("💳 Incentivize switch to automatic payment with bill credit")

    # Paperless billing
    if record.get("PaperlessBilling") == "Yes":
        recs.append("📧 Ensure clear digital billing; unclear e-bills drive churn")

    # Dependents & partner
    if record.get("Partner") == "No" and record.get("Dependents") == "No":
        recs.append("👨‍👩‍👧‍👦 Offer family/partner plans to increase switching cost")

    if not recs:
        recs.append("📞 Schedule proactive outreach call to understand customer needs")

    return recs


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="3.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=_model_cache.get("loaded", False),
        uptime_seconds=round(time.time() - _metrics["uptime_start"], 1),
    )


@app.get("/")
def home() -> FileResponse:
    html_path = os.path.join(_frontend_dir, "index.html") if os.path.isdir(_frontend_dir) else "frontend/index.html"
    return FileResponse(html_path)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus-compatible metrics endpoint."""
    lines = []
    for key, value in _metrics.items():
        if key == "uptime_start":
            continue
        lines.append(f"{key} {value}")
    lines.append(f"uptime_seconds {round(time.time() - _metrics['uptime_start'], 1)}")
    return "\n".join(lines) + "\n"


@app.get("/api/stats")
def api_stats():
    """Return prediction statistics for the dashboard."""
    total = _metrics["prediction_requests_total"]
    positive = _metrics["prediction_churn_positive_total"]
    negative = _metrics["prediction_churn_negative_total"]
    total_preds = positive + negative
    churn_rate = round((positive / total_preds * 100), 1) if total_preds > 0 else 0
    avg_latency = round(
        (_metrics["prediction_latency_seconds_sum"] / total * 1000) if total > 0 else 0, 1
    )
    return {
        "total_predictions": total_preds,
        "churn_count": positive,
        "stay_count": negative,
        "churn_rate": churn_rate,
        "total_requests": total,
        "avg_latency_ms": avg_latency,
        "models_loaded": _model_cache.get("loaded", False),
    }


@app.get("/api/model-info")
def model_info():
    """Return model metadata for the dashboard."""
    if not _model_cache.get("loaded"):
        return {"loaded": False, "message": "Models not loaded — run training first"}

    grid = _model_cache.get("grid")
    bayes = _model_cache.get("bayesian")

    def _model_meta(model, name):
        info = {"name": name}
        if hasattr(model, "n_estimators"):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, "max_depth"):
            info["max_depth"] = model.max_depth
        if hasattr(model, "n_features_in_"):
            info["n_features"] = model.n_features_in_
        return info

    return {
        "loaded": True,
        "grid": _model_meta(grid, "GridSearch RF") if grid else None,
        "bayesian": _model_meta(bayes, "Bayesian RF") if bayes else None,
        "feature_count": len(_model_cache.get("feature_columns") or []),
    }


@app.post("/v1/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.time()
    _metrics["prediction_requests_total"] += 1

    if not _model_cache.get("loaded"):
        _metrics["prediction_errors_total"] += 1
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training pipeline first: python scripts/train_pipeline.py --skip-shap",
        )

    try:
        # Validate input
        issues = validate_inference_input(payload.records)
        if issues:
            logger.warning("Input validation issues: %s", issues)

        grid_model = _model_cache["grid"]
        bayes_model = _model_cache["bayesian"]
        scaler = _model_cache["scaler"]
        feature_columns = _model_cache["feature_columns"]

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

        # Risk levels
        risk_levels = [_get_risk_level(p) for p in proba]

        # Recommendations
        recommendations = [
            _get_recommendations(rec, prob, pr)
            for rec, prob, pr in zip(payload.records, proba, pred)
        ]

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
            risk_levels=risk_levels,
            recommendations=recommendations,
        )

        elapsed = time.time() - start
        _metrics["prediction_latency_seconds_sum"] += elapsed
        logger.info(
            "Prediction served in %.3fs (%d records, model=%s)",
            elapsed, len(pred), payload.model_type,
        )
        return response

    except HTTPException:
        raise
    except Exception as exc:
        _metrics["prediction_errors_total"] += 1
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/reload-models")
def reload_models():
    """Hot-reload models without restarting the server."""
    _load_models_into_cache()
    loaded = _model_cache.get("loaded", False)
    return {"status": "ok" if loaded else "failed", "models_loaded": loaded}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
