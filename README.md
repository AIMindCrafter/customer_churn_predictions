# Churn Prediction — Production MLOps

A professional, modular ML pipeline for customer churn prediction with full MLOps: MLflow experiment tracking & model registry, CI/CD, data validation, drift detection, Docker deployment, and a production UI.

## 📁 Project Structure

```
churn_prediction/
├── src/                                # Core ML modules
│   ├── __init__.py
│   ├── config.py                      # Environment-configurable settings
│   ├── logger.py                      # Structured logging (JSON + human)
│   ├── data_loader.py                 # CSV loading, feature encoding
│   ├── preprocessing.py               # Outlier removal, SMOTE, scaling
│   ├── models.py                      # Training, MLflow registry integration
│   ├── evaluation.py                  # Metrics, confusion matrices, ROC
│   ├── explainability.py              # SHAP feature importance
│   ├── validation.py                  # Data schema validation
│   ├── monitoring.py                  # Prediction logging (JSONL)
│   ├── drift.py                       # Statistical drift detection
│   ├── api.py                         # FastAPI inference service
│   └── utils.py                       # Helpers
├── scripts/
│   ├── train_pipeline.py              # CLI training with MLflow registration
│   ├── predict.py                     # Batch inference script
│   ├── check_drift.py                 # Drift detection CLI
│   └── run.sh                         # Quick pipeline runner
├── tests/
│   └── test_pipeline.py               # Pytest test suite
├── frontend/
│   ├── index.html                     # Dashboard UI
│   ├── styles.css                     # Dark-mode design system
│   └── app.js                         # Client-side logic
├── .github/workflows/
│   └── ci.yml                         # GitHub Actions CI/CD
├── main.py                            # Pipeline orchestration
├── Dockerfile                         # Container build (with healthcheck)
├── docker-compose.yml                 # API + MLflow server
├── requirements.txt                   # All dependencies
├── .env.example                       # Environment variables template
├── .gitignore
└── .dvcignore                         # DVC data tracking config
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete training pipeline
python main.py

# Or use the CLI with options
python scripts/train_pipeline.py --register --skip-shap
```

## 🐳 Docker Deployment (API + MLflow Server)

```bash
# Start everything
docker compose up --build

# Services:
#   - API:    http://localhost:8000
#   - MLflow: http://localhost:5000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with version + timestamp |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/` | Frontend dashboard |
| `POST` | `/predict` | Single/batch prediction |
| `POST` | `/v1/predict` | Versioned prediction endpoint |

### Test prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "bayesian",
    "records": [{
      "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
      "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
      "MultipleLines": "No phone service", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "OnlineBackup": "No",
      "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "Yes", "StreamingMovies": "Yes",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check"
    }]
  }'
```

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Lint
ruff check src/ scripts/ tests/ main.py
```

## 📊 MLflow Experiment Tracking

```bash
# Start MLflow UI (standalone)
mlflow ui --port 5000

# Or use docker compose (starts automatically)
docker compose up mlflow
```

**Features:**
- Automatic experiment logging (params, metrics, models)
- Model Registry with staging/production stages
- Plot artifacts (confusion matrix, ROC curves)
- Git commit hash tracking

## 🔍 Data Drift Detection

```bash
# Check drift against training data reference
python scripts/check_drift.py --n-recent 200 --threshold 2.0

# Save report
python scripts/check_drift.py --output drift_report.json
```

## 🏗️ MLOps Architecture

```
┌─────────────────────────────────────────────────┐
│                   CI/CD Pipeline                 │
│  Lint → Test → Build Docker → Deploy             │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Data     │───▶│ Training │───▶│ MLflow   │   │
│  │ Validation│    │ Pipeline │    │ Registry │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│                                       │           │
│                                       ▼           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Drift   │◀───│Prediction│◀───│ FastAPI  │   │
│  │Detection │    │ Logging  │    │  Server  │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│                                                   │
├─────────────────────────────────────────────────┤
│              Monitoring & Metrics                 │
│  /metrics endpoint · Prediction logs · Alerts     │
└─────────────────────────────────────────────────┘
```

## ✅ Production Standards

- **Structured Logging** — JSON-formatted logs for aggregation tools
- **Data Validation** — Schema checks for training and inference data
- **MLflow Integration** — Experiment tracking, model registry, artifact storage
- **Drift Detection** — Statistical comparison against training reference
- **Prediction Monitoring** — JSONL logging of all predictions
- **Prometheus Metrics** — `/metrics` endpoint for monitoring
- **CI/CD** — GitHub Actions with lint, test, Docker build
- **Docker** — Multi-service compose with healthchecks
- **No Data Leakage** — SMOTE only on training data
- **Reproducibility** — Fixed seeds, git hash tracking
- **SHAP Explainability** — Built-in feature importance analysis

## 📝 Environment Variables

See `.env.example` for all configurable settings including:
- `MLFLOW_TRACKING_URI` — MLflow server URL
- `MODELS_DIR` — Model artifact directory
- `LOG_LEVEL` / `LOG_JSON` — Logging configuration
- `DATA_PATH` — Training data location

---

**Status**: ✅ Production Ready | **Version**: 2.0.0
