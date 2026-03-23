"""Data preprocessing: outlier removal, SMOTE, scaling"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.config import SMOTE_RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)


def remove_outliers_iqr(X, y, threshold=1.5):
    """Remove outliers using IQR method"""
    X_copy = X.copy()
    mask = np.ones(len(X_copy), dtype=bool)

    for col in X_copy.select_dtypes(include=[np.number]).columns:
        Q1 = X_copy[col].quantile(0.25)
        Q3 = X_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        mask &= (X_copy[col] >= lower) & (X_copy[col] <= upper)

    X_clean = X_copy[mask]
    y_clean = y[mask]
    removed = len(X_copy) - len(X_clean)
    logger.info("Outliers removed: %d rows (%.1f%%)", removed, removed / len(X_copy) * 100)
    return X_clean, y_clean


def apply_smote(X_train, y_train):
    """Apply SMOTE for class balancing"""
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    logger.info("SMOTE applied: %s → %s", X_train.shape, X_train_sm.shape)
    return X_train_sm, y_train_sm


def scale_data(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Data scaled with StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(X_train, X_test, y_train, y_test):
    """Complete preprocessing pipeline"""
    # Remove outliers from training data
    X_train_clean, y_train_clean = remove_outliers_iqr(X_train, y_train)

    # Apply SMOTE to training data only
    X_train_sm, y_train_sm = apply_smote(X_train_clean, y_train_clean)

    # Convert to DataFrame to preserve column names
    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

    # Scale all data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_sm, X_test)

    return X_train_scaled, X_test_scaled, y_train_sm, y_test, scaler
