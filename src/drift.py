"""Data and model drift detection using statistical methods.

Compares incoming prediction data distributions against the training
data reference profile to detect feature drift and concept drift.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from src.config import DRIFT_REFERENCE_PATH, PREDICTION_LOG_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def create_reference_profile(X_train: pd.DataFrame, save_path: str | None = None) -> dict:
    """Create a statistical profile of the training data for drift comparison.

    Args:
        X_train: Training features DataFrame.
        save_path: Where to save the profile. Uses config default if None.

    Returns:
        Dictionary with per-column statistics.
    """
    save_path = save_path or DRIFT_REFERENCE_PATH
    profile = {}

    for col in X_train.columns:
        col_data = X_train[col]
        if np.issubdtype(col_data.dtype, np.number):
            profile[col] = {
                "type": "numeric",
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q50": float(col_data.quantile(0.50)),
                "q75": float(col_data.quantile(0.75)),
            }
        else:
            value_counts = col_data.value_counts(normalize=True).to_dict()
            profile[col] = {
                "type": "categorical",
                "distribution": {str(k): float(v) for k, v in value_counts.items()},
            }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(profile, save_path)
    logger.info("Reference profile saved to %s (%d features)", save_path, len(profile))
    return profile


def load_reference_profile(path: str | None = None) -> dict:
    """Load a previously saved reference profile."""
    path = path or DRIFT_REFERENCE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference profile not found at {path}")
    return joblib.load(path)


def detect_drift(current_data: pd.DataFrame, reference: dict | None = None,
                 threshold: float = 2.0) -> dict:
    """Compare current data against reference profile.

    For numeric features, flags drift when the mean shifts by more than
    `threshold` standard deviations from the reference mean.

    For categorical features, flags drift when a new category appears or
    an existing category's proportion changes by more than 20%.

    Args:
        current_data: DataFrame of recent prediction inputs.
        reference: Reference profile dict. Loads from disk if None.
        threshold: Number of std devs for numeric drift detection.

    Returns:
        Dict with 'drifted_features', 'total_features', 'drift_detected',
        and per-feature details.
    """
    if reference is None:
        reference = load_reference_profile()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": 0,
        "drifted_features": [],
        "details": {},
    }

    for col, ref_stats in reference.items():
        if col not in current_data.columns:
            continue

        results["total_features"] += 1
        col_data = current_data[col]

        if ref_stats["type"] == "numeric":
            current_mean = float(col_data.mean())
            ref_mean = ref_stats["mean"]
            ref_std = ref_stats["std"]

            if ref_std > 0:
                z_score = abs(current_mean - ref_mean) / ref_std
            else:
                z_score = 0.0

            drifted = z_score > threshold
            results["details"][col] = {
                "type": "numeric",
                "ref_mean": ref_mean,
                "current_mean": current_mean,
                "z_score": round(z_score, 3),
                "drifted": drifted,
            }
        else:
            ref_dist = ref_stats["distribution"]
            current_dist = col_data.value_counts(normalize=True).to_dict()

            new_categories = set(str(k) for k in current_dist.keys()) - set(ref_dist.keys())
            max_shift = 0.0
            for cat, ref_prop in ref_dist.items():
                cur_prop = current_dist.get(cat, 0.0)
                max_shift = max(max_shift, abs(cur_prop - ref_prop))

            drifted = bool(new_categories) or max_shift > 0.20
            results["details"][col] = {
                "type": "categorical",
                "new_categories": list(new_categories),
                "max_proportion_shift": round(max_shift, 3),
                "drifted": drifted,
            }

        if drifted:
            results["drifted_features"].append(col)

    results["drift_detected"] = len(results["drifted_features"]) > 0

    if results["drift_detected"]:
        logger.warning("Data drift detected in %d/%d features: %s",
                        len(results["drifted_features"]),
                        results["total_features"],
                        results["drifted_features"])
    else:
        logger.info("No data drift detected (%d features checked)",
                     results["total_features"])

    return results


def load_recent_predictions(n_recent: int = 100) -> pd.DataFrame:
    """Load the most recent prediction inputs from the log file.

    Args:
        n_recent: Number of most recent prediction batches to load.

    Returns:
        DataFrame of recent prediction inputs.
    """
    if not os.path.exists(PREDICTION_LOG_PATH):
        raise FileNotFoundError(f"Prediction log not found: {PREDICTION_LOG_PATH}")

    records = []
    with open(PREDICTION_LOG_PATH) as f:
        lines = f.readlines()

    for line in lines[-n_recent:]:
        try:
            entry = json.loads(line)
            records.extend(entry.get("inputs", []))
        except json.JSONDecodeError:
            continue

    if not records:
        raise ValueError("No prediction records found in log")

    return pd.DataFrame(records)


def run_drift_check(n_recent: int = 100, threshold: float = 2.0) -> dict:
    """End-to-end drift detection from prediction logs.

    Args:
        n_recent: Number of recent prediction batches to analyze.
        threshold: Drift sensitivity threshold.

    Returns:
        Drift detection results dict.
    """
    recent_data = load_recent_predictions(n_recent)
    return detect_drift(recent_data, threshold=threshold)
