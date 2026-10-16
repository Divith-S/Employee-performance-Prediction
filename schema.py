"""Shared feature schema and input validation."""

from pathlib import Path

import pandas as pd

TARGET = "PerformanceLevel"
FEATURES = (
    "Department",
    "JobRole",
    "EducationLevel",
    "YearsAtCompany",
    "TrainingHours",
    "ProjectsCompleted",
    "AttendanceRate",
    "ManagerRating",
    "PeerFeedbackScore",
    "MonthlyHours",
    "PromotionLast2Years",
)
NUMERIC_FEATURES = (
    "YearsAtCompany",
    "TrainingHours",
    "ProjectsCompleted",
    "AttendanceRate",
    "ManagerRating",
    "PeerFeedbackScore",
    "MonthlyHours",
)
CATEGORICAL_FEATURES = tuple(
    feature for feature in FEATURES if feature not in NUMERIC_FEATURES
)


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only model features in the expected order, or explain what's missing."""
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input data has no rows")
    return frame.loc[:, FEATURES].copy()


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled CSV and validate its feature and target columns."""
    frame = pd.read_csv(path)
    features = validate_features(frame)
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    target = frame[TARGET]
    if target.isna().any() or target.astype(str).str.strip().eq("").any():
        raise ValueError(f"Target column {TARGET} contains missing or blank values")
    if target.nunique() < 2:
        raise ValueError("Training data must contain at least two target classes")
    return features, target.astype(str)
