"""Data loading and basic preparation"""

import pandas as pd
import numpy as np
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(path=None):
    """Load raw data"""
    path = path or DATA_PATH
    df = pd.read_csv(path)
    logger.info("Data loaded: %s", df.shape)
    return df


def prepare_features(df):
    """Convert categorical to numeric, separate features and target"""
    df_copy = df.copy()

    # Target column
    target_col = 'Churn'
    has_target = target_col in df_copy.columns
    if has_target:
        df_copy[target_col] = df_copy[target_col].map({'Yes': 1, 'No': 0})

    # Encode binary columns
    binary_cols = df_copy.select_dtypes(include=['object']).columns
    for col in binary_cols:
        if df_copy[col].nunique() == 2:
            mapping = {val: i for i, val in enumerate(df_copy[col].unique())}
            df_copy[col] = df_copy[col].map(mapping)

    # One-hot encode categorical columns (>2 categories)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)

    y = df_copy[target_col] if has_target else None
    X = df_copy.drop(columns=[target_col]) if has_target else df_copy

    logger.info("Features prepared: %s", X.shape)
    return X, y


def split_data(X, y):
    """Train-test split"""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
