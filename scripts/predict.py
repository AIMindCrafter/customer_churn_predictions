"""Inference script - Make predictions with trained models"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import load_feature_columns, load_models, load_model_from_mlflow
from src.logger import get_logger

logger = get_logger(__name__)


def predict_churn(data_path, model_type='bayesian', mlflow_model_uri=None):
    """
    Predict churn for new data

    Args:
        data_path: Path to CSV with new customer data
        model_type: 'bayesian' or 'grid'
        mlflow_model_uri: Optional MLflow model URI (e.g., runs:/<run_id>/model)
    """

    # Load new data
    df_new = pd.read_csv(data_path)
    logger.info("Loaded %d new records", len(df_new))

    # Prepare inference features
    if 'Churn' in df_new.columns:
        df_new = df_new.drop(columns=['Churn'])
    X_new = pd.get_dummies(df_new, drop_first=True)

    # Load local models/scaler
    grid_model, bayes_model, scaler = load_models()

    # Align feature order to training schema
    feature_columns = load_feature_columns()
    if feature_columns is not None:
        X_new = X_new.reindex(columns=feature_columns, fill_value=0)

    # Scale new data
    X_new_scaled = scaler.transform(X_new)

    # Select model
    if mlflow_model_uri:
        model = load_model_from_mlflow(mlflow_model_uri)
        model_label = 'mlflow'
    else:
        model = bayes_model if model_type == 'bayesian' else grid_model
        model_label = model_type

    # Make predictions
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'churn_probability': probabilities,
        'risk_level': ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.4 else 'LOW' for p in probabilities]
    })

    # Save results
    output_path = f"predictions_{model_label}.csv"
    results.to_csv(output_path, index=False)

    logger.info("Predictions saved to %s", output_path)
    logger.info("Summary — High: %d | Medium: %d | Low: %d",
                (results['churn_probability'] > 0.7).sum(),
                ((results['churn_probability'] > 0.4) & (results['churn_probability'] <= 0.7)).sum(),
                (results['churn_probability'] <= 0.4).sum())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Make predictions with trained models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with new data')
    parser.add_argument('--model', type=str, default='bayesian',
                        choices=['bayesian', 'grid'], help='Model to use')
    parser.add_argument('--mlflow-model-uri', type=str, default=None,
                        help='Optional MLflow model URI, e.g. runs:/<run_id>/model')

    args = parser.parse_args()

    predict_churn(args.data, args.model, args.mlflow_model_uri)
