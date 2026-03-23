"""SHAP feature importance and explainability analysis"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.logger import get_logger

logger = get_logger(__name__)


def _import_shap():
    """Import SHAP library, install if needed."""
    try:
        import shap
        return shap
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])
        import shap
        return shap


def get_shap_values(model, X_data, model_name="Model"):
    """Generate SHAP values for a model"""
    shap = _import_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    logger.info("SHAP values generated for %s", model_name)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_data, plot_type="bar", title="Feature Importance", save_path=None):
    """Plot SHAP summary visualization"""
    shap = _import_shap()
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, plot_type=plot_type, show=False)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP plot saved to %s", save_path)
    plt.close(fig)
    return fig


def compare_feature_importance(shap_grid, shap_bayes, X_data, feature_names):
    """Compare SHAP feature importance between two models"""
    mean_shap_grid = np.abs(shap_grid).mean(axis=0)
    mean_shap_bayes = np.abs(shap_bayes).mean(axis=0)

    top_features = pd.DataFrame({
        'Feature': feature_names,
        'GridSearch': mean_shap_grid,
        'Bayesian': mean_shap_bayes
    })
    top_features['Average'] = (top_features['GridSearch'] + top_features['Bayesian']) / 2
    top_features = top_features.sort_values('Average', ascending=False)

    logger.info("Top 10 features by SHAP importance:\n%s",
                top_features.head(10).to_string(index=False))

    return top_features


def shap_analysis_pipeline(grid_model, bayes_model, X_test_scaled, feature_names):
    """Complete SHAP analysis"""
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    _, shap_grid = get_shap_values(grid_model, X_test_df, "GridSearch Model")
    _, shap_bayes = get_shap_values(bayes_model, X_test_df, "Bayesian Model")

    logger.info("--- GridSearch Model SHAP Analysis ---")
    plot_shap_summary(shap_grid, X_test_df, plot_type="bar",
                      title="GridSearch - Feature Importance (SHAP)")
    plot_shap_summary(shap_grid, X_test_df, plot_type="summary",
                      title="GridSearch - Feature Contributions (SHAP)")

    logger.info("--- Bayesian Model SHAP Analysis ---")
    plot_shap_summary(shap_bayes, X_test_df, plot_type="bar",
                      title="Bayesian - Feature Importance (SHAP)")
    plot_shap_summary(shap_bayes, X_test_df, plot_type="summary",
                      title="Bayesian - Feature Contributions (SHAP)")

    compare_feature_importance(shap_grid, shap_bayes, X_test_df, feature_names)
