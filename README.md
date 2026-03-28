# ChurnGuard — Production MLOps Customer Churn Prediction

A professional, production-grade ML pipeline for customer churn prediction with full MLOps: model preloading, batch predictions, AI-driven retention recommendations, real-time dashboard, MLflow tracking, CI/CD, Docker deployment, and data drift detection.

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
│   ├── api.py                         # FastAPI inference service (v3.0)
│   └── utils.py                       # Helpers
├── scripts/
│   ├── train_pipeline.py              # CLI training with MLflow registration
│   ├── predict.py                     # Batch inference script
│   ├── check_drift.py                 # Drift detection CLI
│   └── run.sh                         # Quick start (auto-train + serve)
├── tests/
│   └── test_pipeline.py               # Pytest test suite
├── frontend/
│   ├── index.html                     # Premium glassmorphism dashboard
│   ├── styles.css                     # Dark-mode design system
│   └── app.js                         # Client-side logic + batch CSV
├── .github/workflows/
│   └── ci.yml                         # GitHub Actions CI/CD
├── main.py                            # Pipeline orchestration
├── Dockerfile                         # Container build (with healthcheck)
├── docker-compose.yml                 # API + MLflow server
├── requirements.txt                   # All dependencies
├── .env                               # Environment variables
├── .env.example                       # Environment variables template
├── .gitignore
└── .dvcignore                         # DVC data tracking config
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models + Start server (auto-detects if training needed)
bash scripts/run.sh

# Or manually:
python scripts/train_pipeline.py --skip-shap   # Train (~5 min)
python -m uvicorn src.api:app --reload          # Start server
```

Open **http://localhost:8000** to access the dashboard.

## 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| **Single Prediction** | Fill customer profile form, click predict |
| **Batch CSV Upload** | Upload CSV file for multi-customer predictions |
| **Risk Gauge** | Animated circular gauge showing churn probability |
| **Retention Recommendations** | AI-generated strategies to reduce churn |
| **Live Stats** | Real-time prediction counters and churn rate |
| **Model Info** | View loaded model metadata |
| **Prediction History** | Session-persistent history with clear option |
| **Export CSV** | Download batch results as CSV |

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
| `GET` | `/health` | Health check with model status + uptime |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/` | Frontend dashboard |
| `GET` | `/api/stats` | Prediction statistics for dashboard |
| `GET` | `/api/model-info` | Model metadata |
| `POST` | `/predict` | Single/batch prediction with recommendations |
| `POST` | `/v1/predict` | Versioned prediction endpoint |
| `POST` | `/reload-models` | Hot-reload models without restart |

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

## 🔍 Data Drift Detection

```bash
# Check drift against training data reference
python scripts/check_drift.py --n-recent 200 --threshold 2.0
```

## ✅ Production Standards

- **Model Preloading** — Models loaded once at startup, cached in memory
- **CORS Enabled** — Cross-origin access for decoupled frontends
- **Retention Recommendations** — AI-generated strategies for at-risk customers
- **Batch Predictions** — CSV upload for bulk processing
- **Structured Logging** — JSON-formatted logs for aggregation tools
- **Data Validation** — Schema checks for training and inference data
- **MLflow Integration** — Experiment tracking, model registry, artifact storage
- **Drift Detection** — Statistical comparison against training reference
- **Prediction Monitoring** — JSONL logging of all predictions
- **Prometheus Metrics** — `/metrics` endpoint for monitoring
- **CI/CD** — GitHub Actions with lint, test, Docker build
- **Docker** — Multi-service compose with healthchecks
- **Hot Reload** — Reload models without server restart

---

**Status**: ✅ Production Ready | **Version**: 3.0.0
