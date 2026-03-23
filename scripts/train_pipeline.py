"""Reproducible CLI training pipeline with full MLflow integration.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --data path/to/data.csv --register
    python scripts/train_pipeline.py --skip-shap --n-iter 10
"""

import argparse
import os
import sys
import subprocess

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.logger import get_logger
from src.data_loader import load_data, prepare_features, split_data
from src.preprocessing import preprocess_pipeline
from src.validation import validate_dataframe
from src.models import (
    train_grid_search,
    train_bayesian_search,
    save_models,
    setup_mlflow,
    log_search_run_to_mlflow,
)
from src.evaluation import compare_models, plot_confusion_matrices, plot_roc_curves
from src.explainability import shap_analysis_pipeline
from src.config import PLOTS_DIR

logger = get_logger(__name__)


def _git_commit_hash() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def run_pipeline(
    data_path: str | None = None,
    register_model: bool = False,
    skip_shap: bool = False,
    n_iter: int | None = None,
):
    """Run the full training pipeline.

    Args:
        data_path: Path to CSV data file. Uses config default if None.
        register_model: Register trained models in MLflow Model Registry.
        skip_shap: Skip SHAP analysis (faster training).
        n_iter: Override Bayesian optimization iterations.
    """

    git_hash = _git_commit_hash()
    logger.info("=== Training Pipeline Started (git: %s) ===", git_hash)

    # 1. Load & Validate Data
    logger.info("Step 1/7: Loading & validating data")
    df = load_data(data_path)
    issues = validate_dataframe(df, raise_on_error=False)
    if issues:
        logger.warning("Data has %d validation issues — proceeding with caution", len(issues))

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Class distribution — Train: %s", dict(y_train.value_counts()))

    # 2. Preprocess
    logger.info("Step 2/7: Preprocessing")
    X_train_scaled, X_test_scaled, y_train_sm, y_test, scaler = preprocess_pipeline(
        X_train, X_test, y_train, y_test
    )

    # 3. Train Models
    logger.info("Step 3/7: Training models")
    grid_search, grid_time = train_grid_search(X_train_scaled, y_train_sm)

    if n_iter:
        import src.config
        src.config.BAYES_N_ITER = n_iter
    bayes_search, bayes_time = train_bayesian_search(X_train_scaled, y_train_sm)

    # 4. Evaluate
    logger.info("Step 4/7: Evaluating models")
    comparison = compare_models(
        grid_search.best_estimator_,
        bayes_search.best_estimator_,
        X_test_scaled, y_test,
    )

    # 5. Save plots
    logger.info("Step 5/7: Generating visualizations")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_confusion_matrices(
        grid_search.best_estimator_, bayes_search.best_estimator_,
        X_test_scaled, y_test,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"),
    )
    plot_roc_curves(
        grid_search.best_estimator_, bayes_search.best_estimator_,
        X_test_scaled, y_test,
        save_path=os.path.join(PLOTS_DIR, "roc_curves.png"),
    )

    # 6. Save model artifacts
    logger.info("Step 6/7: Saving model artifacts")
    save_models(grid_search, bayes_search, scaler, feature_names=X.columns)

    # 7. MLflow logging
    logger.info("Step 7/7: Logging to MLflow")
    if setup_mlflow():
        try:
            import mlflow

            grid_metrics = comparison.iloc[0].to_dict()
            bayes_metrics = comparison.iloc[1].to_dict()
            grid_metrics.pop("Model", None)
            bayes_metrics.pop("Model", None)

            grid_run_id = log_search_run_to_mlflow(
                run_name="gridsearch_random_forest",
                search_obj=grid_search,
                train_time=grid_time,
                test_metrics=grid_metrics,
                X_example=X_test_scaled,
                register_model=register_model,
            )
            bayes_run_id = log_search_run_to_mlflow(
                run_name="bayesian_random_forest",
                search_obj=bayes_search,
                train_time=bayes_time,
                test_metrics=bayes_metrics,
                X_example=X_test_scaled,
                register_model=register_model,
            )

            # Log additional artifacts to the last run
            with mlflow.start_run(run_name="pipeline_artifacts"):
                mlflow.set_tag("git_commit", git_hash)
                mlflow.set_tag("data_rows", str(len(df)))
                mlflow.set_tag("data_columns", str(len(df.columns)))
                mlflow.log_metric("train_rows_after_smote", len(y_train_sm))
                mlflow.log_metric("test_rows", len(y_test))

                # Log plots as artifacts
                cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
                roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
                if os.path.exists(cm_path):
                    mlflow.log_artifact(cm_path, "plots")
                if os.path.exists(roc_path):
                    mlflow.log_artifact(roc_path, "plots")

            logger.info("MLflow runs: grid=%s | bayes=%s", grid_run_id, bayes_run_id)
        except Exception as exc:
            logger.error("MLflow logging failed: %s", exc)

    # 8. SHAP (optional)
    if not skip_shap:
        logger.info("Bonus: SHAP Analysis")
        shap_analysis_pipeline(
            grid_search.best_estimator_,
            bayes_search.best_estimator_,
            X_test_scaled, X.columns,
        )

    logger.info("=== Pipeline Complete (Grid: %.1fs, Bayes: %.1fs) ===",
                grid_time, bayes_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Training Pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV data file")
    parser.add_argument("--register", action="store_true",
                        help="Register models in MLflow Model Registry")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis for faster training")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Override Bayesian optimization iterations")

    args = parser.parse_args()
    run_pipeline(
        data_path=args.data,
        register_model=args.register,
        skip_shap=args.skip_shap,
        n_iter=args.n_iter,
    )
