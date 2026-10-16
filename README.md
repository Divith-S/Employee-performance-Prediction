# Employee Performance Prediction

A reproducible machine-learning project that classifies employee performance as **Needs Improvement**, **Meets Expectations**, or **Exceeds Expectations**. It includes a small exploratory analysis, preprocessing and model training pipeline, batch prediction command, tests, and a ready-to-run synthetic sample dataset.

> **Data honesty:** The included CSV contains 1,200 simulated records generated for this project. It is not real employee or company data, and its model scores are only a software demonstration. Replace it with authorized, appropriately anonymized data before drawing any real-world conclusions.

## What it does

- Summarizes the dataset, missing values, class distribution, and numeric features.
- Imputes missing feature values and one-hot encodes categorical features inside a scikit-learn pipeline.
- Trains a class-balanced Random Forest and reports accuracy, macro F1, per-class metrics, confusion matrix, and leading feature importances.
- Saves a reusable model artifact and predicts classes plus class probabilities for new CSV rows.
- Checks the expected input schema and has automated tests for data validation, training, and inference.

![Exploratory charts from the included synthetic dataset](reports/eda_overview.png)

On the included synthetic dataset, the fixed-seed stratified holdout run records **71.7% accuracy** and **0.638 macro F1** (240 test rows). These are reproducibility figures for generated data only, not evidence of real-world accuracy. See [`reports/metrics.json`](reports/metrics.json) for class-level results and the confusion matrix.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run exploratory analysis, train the model, then predict on the included example:

```bash
python -m employee_performance.analyze
python -m employee_performance.train
python -m employee_performance.predict --input data/new_employees_example.csv
pytest
```

The prediction command writes `reports/predictions.csv`. Training writes `models/performance_model.joblib` and `reports/metrics.json`; analysis writes `reports/eda_summary.json` and `reports/eda_overview.png`. Generated model and image files are intentionally excluded from Git by `.gitignore` and can be recreated with the commands above.

To recreate the sample dataset:

```bash
python scripts/generate_sample_data.py --rows 1200 --seed 2025
```

## Use your own data

Create a CSV with one row per employee. Training requires all model features and a `PerformanceLevel` target column. Feature fields may contain blanks; the pipeline imputes them. The target must be present and contain at least two classes.

| Column | Type | Example |
| --- | --- | --- |
| Department | category | Engineering |
| JobRole | category | Data Analyst |
| EducationLevel | category | Bachelor's |
| YearsAtCompany | number | 3 |
| TrainingHours | number | 24 |
| ProjectsCompleted | number | 5 |
| AttendanceRate | number | 96.5 |
| ManagerRating | number | 4 |
| PeerFeedbackScore | number | 4 |
| MonthlyHours | number | 168 |
| PromotionLast2Years | category | Yes |
| PerformanceLevel | target | Meets Expectations |

`EmployeeID` and other extra columns are ignored by the model. They are retained in prediction output if provided. Use only data you are authorized to process, remove direct identifiers where possible, and confirm that each feature is appropriate for the intended use.

Train/evaluate with a different file and output paths:

```bash
python -m employee_performance.train \
  --data path/to/authorized_training_data.csv \
  --model models/performance_model.joblib \
  --metrics reports/metrics.json
```

Predict with a CSV containing the feature columns above (the target column is not needed):

```bash
python -m employee_performance.predict \
  --input path/to/new_employees.csv \
  --model models/performance_model.joblib \
  --output reports/predictions.csv
```

## Project layout

```text
data/                 Synthetic demo data and prediction example
reports/              Training metrics and EDA summary output
scripts/              Reproducible synthetic-data generator
src/employee_performance/
  analyze.py           Exploratory data analysis
  predict.py           Batch inference CLI
  schema.py            Shared feature definitions and validation
  train.py             Preprocessing, training, evaluation, and model export
tests/                Automated unit and workflow tests
.github/workflows/    GitHub Actions test workflow
```

## Evaluation and responsible use

The random train/test split is stratified and uses a fixed seed for repeatability. Macro F1 is included so each class contributes equally to the summary metric. Synthetic-data scores do not estimate performance on a real workforce. Before any real deployment, use representative historical data, check leakage and subgroup performance, review applicable privacy and employment rules, and have qualified people assess every prediction. This model is an educational decision-support example, not a substitute for human judgment and not a basis for hiring, firing, compensation, or promotion decisions.

## License

MIT. See [LICENSE](LICENSE).
