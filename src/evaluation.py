"""Model evaluation and metrics"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }

    logger.info("%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
                model_name, accuracy, precision, recall, f1, auc)

    return metrics, y_pred, y_pred_proba


def compare_models(grid_model, bayes_model, X_test, y_test):
    """Compare two models side by side"""
    grid_metrics, _, _ = evaluate_model(grid_model, X_test, y_test, "GridSearch RF")
    bayes_metrics, _, _ = evaluate_model(bayes_model, X_test, y_test, "Bayesian RF")

    comparison = pd.DataFrame([grid_metrics, bayes_metrics])
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    return comparison


def plot_confusion_matrices(grid_model, bayes_model, X_test, y_test, save_path=None):
    """Plot confusion matrices for both models"""
    y_pred_grid = grid_model.predict(X_test)
    y_pred_bayes = bayes_model.predict(X_test)

    cm_grid = confusion_matrix(y_test, y_pred_grid)
    cm_bayes = confusion_matrix(y_test, y_pred_bayes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (cm, title) in enumerate([(cm_grid, "GridSearch"), (cm_bayes, "Bayesian")]):
        axes[idx].imshow(cm, cmap='Blues', alpha=0.7)
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, cm[i, j], ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)
    return fig


def plot_roc_curves(grid_model, bayes_model, X_test, y_test, save_path=None):
    """Plot ROC curves for both models"""
    y_pred_proba_grid = grid_model.predict_proba(X_test)[:, 1]
    y_pred_proba_bayes = bayes_model.predict_proba(X_test)[:, 1]

    fpr_grid, tpr_grid, _ = roc_curve(y_test, y_pred_proba_grid)
    fpr_bayes, tpr_bayes, _ = roc_curve(y_test, y_pred_proba_bayes)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr_grid, tpr_grid, label=f'GridSearch (AUC={roc_auc_score(y_test, y_pred_proba_grid):.4f})', linewidth=2)
    plt.plot(fpr_bayes, tpr_bayes, label=f'Bayesian (AUC={roc_auc_score(y_test, y_pred_proba_bayes):.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)
    return fig
