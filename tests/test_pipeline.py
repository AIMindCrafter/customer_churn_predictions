"""Unit and integration tests for the churn prediction project."""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════

class TestConfig:
    def test_config_imports(self):
        from src.config import RANDOM_STATE, TEST_SIZE
        assert RANDOM_STATE == 42
        assert 0 < TEST_SIZE < 1

    def test_config_env_override(self):
        with patch.dict(os.environ, {"DATA_PATH": "/tmp/test.csv"}):
            # Re-import to pick up env
            import importlib
            import src.config
            importlib.reload(src.config)
            assert src.config.DATA_PATH == "/tmp/test.csv"
            # Restore
            importlib.reload(src.config)


# ═══════════════════════════════════════════════════════════
# Data Loader Tests
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV for testing."""
    data = {
        "customerID": ["001", "002", "003", "004", "005"],
        "tenure": [12, 24, 36, 48, 60],
        "MonthlyCharges": [70.5, 80.0, 50.5, 90.0, 60.0],
        "TotalCharges": [846, 1920, 1818, 4320, 3600],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "Yes", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No", "Yes", "No internet service", "No"],
        "TechSupport": ["No", "Yes", "No", "No internet service", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes", "No internet service", "No"],
        "StreamingMovies": ["Yes", "No", "No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check",
        ],
        "Churn": ["Yes", "No", "No", "Yes", "No"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestDataLoader:
    def test_load_data(self, sample_csv):
        from src.data_loader import load_data
        df = load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_prepare_features(self, sample_csv):
        from src.data_loader import load_data, prepare_features
        df = load_data(sample_csv)
        X, y = prepare_features(df)
        assert "Churn" not in X.columns
        assert len(y) == 5
        assert set(y.unique()).issubset({0, 1})

    def test_split_data(self, sample_csv):
        from src.data_loader import load_data, prepare_features, split_data
        df = load_data(sample_csv)
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)


# ═══════════════════════════════════════════════════════════
# Preprocessing Tests
# ═══════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_remove_outliers_iqr(self):
        from src.preprocessing import remove_outliers_iqr
        X = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [5, 6, 7, 8, 9]})
        y = pd.Series([0, 1, 0, 1, 0])
        X_clean, y_clean = remove_outliers_iqr(X, y)
        assert len(X_clean) <= len(X)

    def test_apply_smote(self):
        from src.preprocessing import apply_smote
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series([0] * 80 + [1] * 20)
        X_sm, y_sm = apply_smote(X, y)
        # After SMOTE, classes should be balanced
        assert y_sm.value_counts()[0] == y_sm.value_counts()[1]

    def test_scale_data(self):
        from src.preprocessing import scale_data
        X_train = pd.DataFrame(np.random.rand(50, 3))
        X_test = pd.DataFrame(np.random.rand(20, 3))
        X_train_s, X_test_s, scaler = scale_data(X_train, X_test)
        # Scaled training data should have ~0 mean
        assert abs(X_train_s.mean()) < 0.5


# ═══════════════════════════════════════════════════════════
# Validation Tests
# ═══════════════════════════════════════════════════════════

class TestValidation:
    def test_validate_columns_pass(self):
        from src.validation import validate_columns, REQUIRED_COLUMNS
        df = pd.DataFrame({c: [1] for c in REQUIRED_COLUMNS})
        issues = validate_columns(df)
        assert issues == []

    def test_validate_columns_missing(self):
        from src.validation import validate_columns
        df = pd.DataFrame({"tenure": [1]})
        issues = validate_columns(df)
        assert len(issues) == 1
        assert "Missing columns" in issues[0]

    def test_validate_numeric_ranges_pass(self):
        from src.validation import validate_numeric_ranges
        df = pd.DataFrame({"tenure": [10], "MonthlyCharges": [50.0],
                           "TotalCharges": [500], "SeniorCitizen": [0]})
        issues = validate_numeric_ranges(df)
        assert issues == []

    def test_validate_numeric_ranges_fail(self):
        from src.validation import validate_numeric_ranges
        df = pd.DataFrame({"tenure": [-5], "MonthlyCharges": [50.0],
                           "TotalCharges": [500], "SeniorCitizen": [0]})
        issues = validate_numeric_ranges(df)
        assert any("below" in i for i in issues)

    def test_validate_categories_pass(self):
        from src.validation import validate_categories
        df = pd.DataFrame({"Partner": ["Yes"], "Dependents": ["No"],
                           "Contract": ["One year"]})
        issues = validate_categories(df)
        assert issues == []

    def test_validate_categories_fail(self):
        from src.validation import validate_categories
        df = pd.DataFrame({"Partner": ["Maybe"]})
        issues = validate_categories(df)
        assert len(issues) == 1

    def test_validate_dataframe_raises(self):
        from src.validation import validate_dataframe, DataValidationError
        df = pd.DataFrame({"tenure": [-5]})
        with pytest.raises(DataValidationError):
            validate_dataframe(df)


# ═══════════════════════════════════════════════════════════
# Evaluation Tests
# ═══════════════════════════════════════════════════════════

class TestEvaluation:
    def test_evaluate_model(self):
        from src.evaluation import evaluate_model
        from sklearn.ensemble import RandomForestClassifier
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        metrics, y_pred, y_proba = evaluate_model(model, X, y, "Test")
        assert "Accuracy" in metrics
        assert 0 <= metrics["AUC-ROC"] <= 1


# ═══════════════════════════════════════════════════════════
# Monitoring Tests
# ═══════════════════════════════════════════════════════════

class TestMonitoring:
    def test_log_prediction(self, tmp_path):
        from src import monitoring
        log_path = str(tmp_path / "pred.jsonl")
        with patch.object(monitoring, "PREDICTION_LOG_PATH", log_path):
            monitoring.log_prediction(
                input_records=[{"tenure": 12}],
                predictions=[1],
                probabilities=[0.85],
                model_type="bayesian",
            )
        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert entry["predictions"] == [1]
        assert entry["model_type"] == "bayesian"


# ═══════════════════════════════════════════════════════════
# API Tests
# ═══════════════════════════════════════════════════════════

class TestAPI:
    def test_health_endpoint(self):
        from src.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_metrics_endpoint(self):
        from src.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text


# ═══════════════════════════════════════════════════════════
# Logger Tests
# ═══════════════════════════════════════════════════════════

class TestLogger:
    def test_get_logger(self):
        from src.logger import get_logger
        log = get_logger("test_logger")
        assert log.name == "test_logger"

    def test_json_formatter(self):
        from src.logger import get_logger
        log = get_logger("test_json", json_format=True)
        assert log.name == "test_json"
