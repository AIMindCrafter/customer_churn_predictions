"""Churn Prediction — Production ML Pipeline

Modules:
    config         - Environment-configurable settings
    logger         - Structured logging (JSON + human-readable)
    data_loader    - CSV loading, feature encoding, train/test split
    preprocessing  - Outlier removal, SMOTE, StandardScaler
    models         - GridSearch + Bayesian training, MLflow integration
    evaluation     - Metrics, confusion matrices, ROC curves
    explainability - SHAP feature importance analysis
    validation     - Data schema validation for training and inference
    monitoring     - Prediction logging for drift detection
    drift          - Statistical drift detection
    api            - FastAPI inference service
"""
