"""Main orchestration script for Churn Prediction Pipeline"""

import os
import sys
from src.utils import print_section
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


def main():
    """Execute complete pipeline"""

    # 1. Load & Prepare Data
    print_section("1. LOADING & PREPARING DATA")
    df = load_data()

    # 1b. Validate raw data
    print_section("1b. VALIDATING RAW DATA")
    validate_dataframe(df, raise_on_error=False)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Preprocessing
    print_section("2. PREPROCESSING DATA")
    X_train_scaled, X_test_scaled, y_train_sm, y_test, scaler = preprocess_pipeline(
        X_train, X_test, y_train, y_test
    )

    # 3. Train Models
    print_section("3. TRAINING MODELS")
    logger.info("Training GridSearch...")
    grid_search, grid_time = train_grid_search(X_train_scaled, y_train_sm)

    logger.info("Training Bayesian Search...")
    bayes_search, bayes_time = train_bayesian_search(X_train_scaled, y_train_sm)

    # 4. Evaluate Models
    print_section("4. MODEL EVALUATION")
    comparison = compare_models(grid_search.best_estimator_,
                                bayes_search.best_estimator_,
                                X_test_scaled, y_test)

    # 5. Visualizations
    print_section("5. VISUALIZATIONS")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_confusion_matrices(grid_search.best_estimator_,
                            bayes_search.best_estimator_,
                            X_test_scaled, y_test,
                            save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"))

    plot_roc_curves(grid_search.best_estimator_,
                    bayes_search.best_estimator_,
                    X_test_scaled, y_test,
                    save_path=os.path.join(PLOTS_DIR, "roc_curves.png"))

    # 6. Save Models
    print_section("6. SAVING MODELS")
    save_models(grid_search, bayes_search, scaler, feature_names=X.columns)

    # 7. MLflow Tracking
    print_section("7. MLFLOW TRACKING")
    if setup_mlflow():
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
            register_model=True,
        )
        bayes_run_id = log_search_run_to_mlflow(
            run_name="bayesian_random_forest",
            search_obj=bayes_search,
            train_time=bayes_time,
            test_metrics=bayes_metrics,
            X_example=X_test_scaled,
            register_model=True,
        )

        logger.info("GridSearch run_id: %s", grid_run_id)
        logger.info("Bayesian run_id: %s", bayes_run_id)

    # 8. SHAP Analysis
    print_section("8. SHAP FEATURE IMPORTANCE ANALYSIS")
    shap_analysis_pipeline(grid_search.best_estimator_,
                           bayes_search.best_estimator_,
                           X_test_scaled,
                           X.columns)

    print_section("✓ PIPELINE COMPLETE")
    logger.info("GridSearch Time: %.1fs | Bayesian Time: %.1fs", grid_time, bayes_time)


if __name__ == "__main__":
    main()
