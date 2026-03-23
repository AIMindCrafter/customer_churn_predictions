"""Configuration file for Churn Prediction project"""

import os

# ── Data Paths ──────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./")

# ── Model Parameters ────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# ── SMOTE Parameters ────────────────────────────────────────
SMOTE_RANDOM_STATE = 42

# ── GridSearch Parameters ────────────────────────────────────
GRID_CV_FOLDS = 5

# ── Bayesian Optimization Parameters ────────────────────────
BAYES_N_ITER = 30
BAYES_CV_FOLDS = 5

# ── Artifact Directories ────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "./artifacts")
PLOTS_DIR = os.getenv("PLOTS_DIR", "./artifacts/plots")
GRID_MODEL_FILE = "best_model_gridsearch_rf.pkl"
BAYES_MODEL_FILE = "best_model_bayesian_rf.pkl"
GRID_PARAMS_FILE = "best_params_gridsearch.pkl"
BAYES_PARAMS_FILE = "best_params_bayesian.pkl"
SCALER_FILE = "scaler.pkl"

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

# ── MLflow Tracking ─────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction_random_forest")

# ── Monitoring ───────────────────────────────────────────────
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "./artifacts/prediction_log.jsonl")
DRIFT_REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "./artifacts/drift_reference.pkl")
