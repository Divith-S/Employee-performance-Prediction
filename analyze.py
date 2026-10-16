"""Create a compact exploratory data analysis summary and chart."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from employee_performance.schema import NUMERIC_FEATURES, TARGET, load_training_data


def analyze(data_path: Path, summary_path: Path, chart_path: Path) -> dict:
    features, target = load_training_data(data_path)
    frame = features.copy()
    frame[TARGET] = target
    summary = {
        "dataset": str(data_path),
        "synthetic_demo_data": "synthetic" in data_path.name.lower(),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "performance_distribution": {
            str(key): int(value) for key, value in target.value_counts().sort_index().items()
        },
        "numeric_summary": json.loads(
            frame.loc[:, list(NUMERIC_FEATURES)].describe().round(2).to_json()
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    counts = target.value_counts().reindex(sorted(target.unique()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#d98570", "#4e79a7", "#55a868"][: len(counts)]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Performance category counts")
    axes[0].set_ylabel("Employees")
    axes[0].tick_params(axis="x", rotation=16)

    grouped = frame.groupby(TARGET, observed=False)["AttendanceRate"].mean()
    grouped = grouped.reindex(counts.index)
    axes[1].bar(grouped.index, grouped.values, color=colors)
    axes[1].set_title("Average attendance by category")
    axes[1].set_ylabel("Attendance rate (%)")
    axes[1].tick_params(axis="x", rotation=16)
    fig.suptitle("Exploratory overview (demo data is synthetic)", fontsize=13)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the labeled employee dataset")
    parser.add_argument("--data", type=Path, default=Path("data/employee_performance_synthetic.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/eda_summary.json"))
    parser.add_argument("--chart", type=Path, default=Path("reports/eda_overview.png"))
    args = parser.parse_args()

    summary = analyze(args.data, args.summary, args.chart)
    print(f"Rows: {summary['rows']}; columns: {summary['columns']}")
    print(f"Class distribution: {summary['performance_distribution']}")
    print(f"Summary saved to: {args.summary}")
    print(f"Chart saved to: {args.chart}")


if __name__ == "__main__":
    main()
