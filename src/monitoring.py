"""Prediction logging for monitoring and drift detection."""

import json
import os
from datetime import datetime, timezone
from src.config import PREDICTION_LOG_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def log_prediction(input_records: list[dict], predictions: list[int],
                   probabilities: list[float], model_type: str) -> None:
    """Append prediction event to JSONL log file.

    Each line is a self-contained JSON object with timestamp, model info,
    input features, and prediction outputs.
    """
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH) or ".", exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "count": len(predictions),
        "inputs": input_records,
        "predictions": predictions,
        "churn_probability": probabilities,
    }

    try:
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.debug("Prediction logged (%d records)", len(predictions))
    except Exception as exc:
        logger.error("Failed to log prediction: %s", exc)
