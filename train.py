"""Train and evaluate the employee performance classifier."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from employee_performance.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_training_data,
)


def build_model(random_state: int) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def train(data_path: Path, model_path: Path, metrics_path: Path, random_state: int) -> dict:
    features, target = load_training_data(data_path)
    counts = target.value_counts()
    if (counts < 2).any():
        raise ValueError("Each target class needs at least two rows for a stratified split")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    model = build_model(random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(target.unique())

    transformed_names = model.named_steps["preprocessing"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top_features = [
        {"feature": str(transformed_names[index]), "importance": float(importances[index])}
        for index in importances.argsort()[::-1][:15]
    ]
    metrics = {
        "dataset": str(data_path),
        "synthetic_demo_data": "synthetic" in data_path.name.lower(),
        "rows": int(len(features)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
        "target": TARGET,
        "classes": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "top_features": top_features,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(FEATURES), "target": TARGET, "classes": labels},
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the performance classifier")
    parser.add_argument("--data", type=Path, default=Path("data/employee_performance_synthetic.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/performance_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.data, args.model, args.metrics, args.random_state)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Model saved to: {args.model}")
    print(f"Metrics saved to: {args.metrics}")


if __name__ == "__main__":
    main()
