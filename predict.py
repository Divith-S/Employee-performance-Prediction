"""Predict performance categories for rows in a CSV file."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from employee_performance.schema import validate_features


def predict(input_path: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    model = payload["model"]
    original = pd.read_csv(input_path)
    features = validate_features(original)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    result = original.copy()
    result["PredictedPerformance"] = predictions
    result["PredictionConfidence"] = probabilities.max(axis=1).round(4)
    for index, label in enumerate(model.classes_):
        result[f"Probability_{label.replace(' ', '_')}"] = probabilities[:, index].round(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict performance categories from a CSV")
    parser.add_argument("--input", type=Path, required=True, help="CSV with all model feature columns")
    parser.add_argument("--model", type=Path, default=Path("models/performance_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args()

    result = predict(args.input, args.model, args.output)
    print(f"Predicted {len(result)} rows; results saved to: {args.output}")


if __name__ == "__main__":
    main()
