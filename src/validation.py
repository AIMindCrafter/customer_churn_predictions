"""Data validation schemas and checks for training and inference data."""

import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

# ── Expected raw data schema ────────────────────────────────

REQUIRED_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLUMNS = [
    "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]

EXPECTED_CATEGORIES = {
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
}

NUMERIC_RANGES = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 50000),
    "SeniorCitizen": (0, 1),
}


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """Check that all required columns are present. Returns list of issues."""
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    return []


def validate_nulls(df: pd.DataFrame, max_null_pct: float = 0.05) -> list[str]:
    """Check for excessive null values."""
    issues = []
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > max_null_pct:
            issues.append(f"Column '{col}' has {null_pct:.1%} nulls (max {max_null_pct:.0%})")
    return issues


def validate_numeric_ranges(df: pd.DataFrame, ranges: dict | None = None) -> list[str]:
    """Check numeric columns are within expected ranges."""
    ranges = ranges or NUMERIC_RANGES
    issues = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        below = (col_numeric < lo).sum()
        above = (col_numeric > hi).sum()
        if below > 0:
            issues.append(f"Column '{col}': {below} values below {lo}")
        if above > 0:
            issues.append(f"Column '{col}': {above} values above {hi}")
    return issues


def validate_categories(df: pd.DataFrame, expected: dict | None = None) -> list[str]:
    """Check categorical columns contain only expected values."""
    expected = expected or EXPECTED_CATEGORIES
    issues = []
    for col, valid in expected.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - valid
        if unexpected:
            issues.append(f"Column '{col}': unexpected values {unexpected}")
    return issues


def validate_dataframe(df: pd.DataFrame, raise_on_error: bool = True) -> list[str]:
    """Run all validation checks on a DataFrame.

    Args:
        df: The DataFrame to validate.
        raise_on_error: If True, raise DataValidationError on first issue set.

    Returns:
        List of issue strings (empty if all checks pass).
    """
    all_issues: list[str] = []
    all_issues.extend(validate_columns(df))
    all_issues.extend(validate_nulls(df))
    all_issues.extend(validate_numeric_ranges(df))
    all_issues.extend(validate_categories(df))

    if all_issues:
        for issue in all_issues:
            logger.warning("Validation: %s", issue)
        if raise_on_error:
            raise DataValidationError(
                f"Data validation failed with {len(all_issues)} issue(s):\n"
                + "\n".join(f"  - {i}" for i in all_issues)
            )
    else:
        logger.info("Data validation passed (%d rows, %d columns)", len(df), len(df.columns))

    return all_issues


def validate_inference_input(records: list[dict]) -> list[str]:
    """Validate a list of dicts (API input) before prediction."""
    df = pd.DataFrame(records)
    # Only validate columns that are present against their expected values
    issues: list[str] = []
    issues.extend(validate_numeric_ranges(df))
    issues.extend(validate_categories(df))
    if issues:
        for issue in issues:
            logger.warning("Inference validation: %s", issue)
    return issues
