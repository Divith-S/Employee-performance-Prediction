import pandas as pd
import pytest

from employee_performance.predict import predict
from employee_performance.schema import FEATURES, TARGET, load_training_data, validate_features
from employee_performance.train import train
from scripts.generate_sample_data import generate_rows


def sample_frame(rows=120):
    return pd.DataFrame(generate_rows(rows, seed=19))


def test_training_data_loads_and_ignores_extra_columns(tmp_path):
    frame = sample_frame()
    frame["UnusedColumn"] = "ignored"
    path = tmp_path / "training.csv"
    frame.to_csv(path, index=False)

    features, target = load_training_data(path)

    assert tuple(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 120


def test_schema_reports_missing_features():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_features(pd.DataFrame({"Department": ["Engineering"]}))


def test_training_rejects_blank_target(tmp_path):
    frame = sample_frame()
    frame.loc[0, TARGET] = None
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="contains missing or blank values"):
        load_training_data(path)


def test_train_and_predict_work_end_to_end(tmp_path):
    frame = sample_frame(180)
    training_path = tmp_path / "training.csv"
    input_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "predictions.csv"
    frame.to_csv(training_path, index=False)
    frame.drop(columns=[TARGET]).head(5).to_csv(input_path, index=False)

    metrics = train(training_path, model_path, metrics_path, random_state=7)
    results = predict(input_path, model_path, output_path)

    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["rows"] == 180
    assert len(results) == 5
    assert "PredictedPerformance" in results
    assert "PredictionConfidence" in results
    assert output_path.exists()
