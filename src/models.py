"""Model training and hyperparameter tuning"""

import time
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from src.config import (
    GRID_CV_FOLDS,
    BAYES_N_ITER,
    BAYES_CV_FOLDS,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None


class IdentityScaler:
    """Fallback scaler used when scaler artifact is unavailable."""

    def transform(self, X):
        return X


def train_grid_search(X_train, y_train):
    """GridSearchCV for Random Forest hyperparameter tuning"""
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=GRID_CV_FOLDS, n_jobs=-1, verbose=1)

    start = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start

    logger.info("GridSearch complete in %.1fs — Best CV Score: %.4f",
                grid_time, grid_search.best_score_)
    return grid_search, grid_time


def train_bayesian_search(X_train, y_train):
    """BayesSearchCV for Random Forest hyperparameter tuning"""
    from skopt import BayesSearchCV
    from skopt.space import Integer, Categorical

    search_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 25),
        'max_features': Categorical(['sqrt', 'log2']),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5)
    }

    rf = RandomForestClassifier(random_state=42)
    bayes_search = BayesSearchCV(
        rf, search_space, n_iter=BAYES_N_ITER, cv=BAYES_CV_FOLDS,
        n_jobs=-1, verbose=1, random_state=42
    )

    start = time.time()
    bayes_search.fit(X_train, y_train)
    bayes_time = time.time() - start

    logger.info("Bayesian Search complete in %.1fs — Best CV Score: %.4f",
                bayes_time, bayes_search.best_score_)
    return bayes_search, bayes_time


def save_models(grid_search, bayes_search, scaler, output_dir=None, feature_names=None):
    """Save trained models and scaler"""
    output_dir = output_dir or MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(grid_search.best_estimator_, os.path.join(output_dir, "best_model_gridsearch_rf.pkl"))
    joblib.dump(bayes_search.best_estimator_, os.path.join(output_dir, "best_model_bayesian_rf.pkl"))
    joblib.dump(grid_search.best_params_, os.path.join(output_dir, "best_params_gridsearch.pkl"))
    joblib.dump(bayes_search.best_params_, os.path.join(output_dir, "best_params_bayesian.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    if feature_names is not None:
        joblib.dump(list(feature_names), os.path.join(output_dir, "feature_columns.pkl"))
    logger.info("Models saved to %s", output_dir)


def load_models(output_dir=None):
    """Load trained models and scaler"""
    output_dir = output_dir or MODELS_DIR
    grid_path = os.path.join(output_dir, "best_model_gridsearch_rf.pkl")
    bayes_path = os.path.join(output_dir, "best_model_bayesian_rf.pkl")
    final_path = os.path.join(output_dir, "final_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    # Also check root-level paths for backward compat
    root_grid = "./best_model_gridsearch_rf.pkl"
    root_bayes = "./best_model_bayesian_rf.pkl"
    root_final = "./final_model.pkl"
    root_scaler = "./scaler.pkl"

    if os.path.exists(grid_path):
        grid_model = joblib.load(grid_path)
    elif os.path.exists(root_grid):
        grid_model = joblib.load(root_grid)
    elif os.path.exists(final_path):
        grid_model = joblib.load(final_path)
    elif os.path.exists(root_final):
        grid_model = joblib.load(root_final)
    else:
        raise FileNotFoundError(f"Missing model artifact: {grid_path}")

    if os.path.exists(bayes_path):
        bayes_model = joblib.load(bayes_path)
    elif os.path.exists(root_bayes):
        bayes_model = joblib.load(root_bayes)
    elif os.path.exists(final_path):
        bayes_model = joblib.load(final_path)
    elif os.path.exists(root_final):
        bayes_model = joblib.load(root_final)
    else:
        raise FileNotFoundError(f"Missing model artifact: {bayes_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    elif os.path.exists(root_scaler):
        scaler = joblib.load(root_scaler)
    else:
        scaler = IdentityScaler()
        logger.warning("scaler.pkl not found. Using IdentityScaler fallback.")

    logger.info("Models loaded from %s", output_dir)
    return grid_model, bayes_model, scaler


def load_feature_columns(output_dir=None):
    """Load training feature column order if available."""
    output_dir = output_dir or MODELS_DIR
    feature_path = os.path.join(output_dir, "feature_columns.pkl")
    root_path = "./feature_columns.pkl"
    if os.path.exists(feature_path):
        return joblib.load(feature_path)
    if os.path.exists(root_path):
        return joblib.load(root_path)
    return None


def _to_builtin_dict(params):
    """Convert numpy/scalar values into Python-native values for MLflow logging."""
    clean = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def setup_mlflow():
    """Configure MLflow tracking destination and experiment."""
    if mlflow is None:
        logger.warning("MLflow is not installed. Skipping MLflow tracking.")
        return False

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow configured — URI: %s | Experiment: %s",
                MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    return True


def log_search_run_to_mlflow(run_name, search_obj, train_time, test_metrics,
                             X_example=None, register_model=False):
    """Log search parameters, metrics, and best model as an MLflow run.

    Args:
        register_model: If True, register the model in MLflow Model Registry.
    """
    if mlflow is None:
        return None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_to_builtin_dict(search_obj.best_params_))
        mlflow.log_metric("cv_best_score", float(search_obj.best_score_))
        mlflow.log_metric("train_time_sec", float(train_time))

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

        input_example = None
        if X_example is not None and len(X_example) > 0:
            input_example = X_example[:5]

        registered_name = f"churn-{run_name}" if register_model else None

        mlflow.sklearn.log_model(
            sk_model=search_obj.best_estimator_,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=registered_name,
        )

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run logged: %s (run_id=%s)", run_name, run_id)
        return run_id


def load_model_from_mlflow(model_uri):
    """Load a scikit-learn model from MLflow model URI."""
    if mlflow is None:
        raise ImportError("MLflow is not installed. Install mlflow first.")
    return mlflow.sklearn.load_model(model_uri)
