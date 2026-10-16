"""Generate clearly labeled synthetic employee data for demos and tests."""

import argparse
import csv
import random
from pathlib import Path


DEPARTMENTS = ["Engineering", "Sales", "Human Resources", "Finance", "Operations", "Marketing"]
ROLES = {
    "Engineering": ["Software Engineer", "Data Analyst", "QA Engineer"],
    "Sales": ["Sales Executive", "Account Manager", "Sales Analyst"],
    "Human Resources": ["HR Generalist", "Recruiter", "People Analyst"],
    "Finance": ["Financial Analyst", "Accountant", "Payroll Specialist"],
    "Operations": ["Operations Analyst", "Project Coordinator", "Team Lead"],
    "Marketing": ["Marketing Analyst", "Content Specialist", "Campaign Manager"],
}
FIELDS = [
    "EmployeeID",
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
    "PerformanceLevel",
]


def generate_rows(rows: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    dataset = []
    for index in range(1, rows + 1):
        department = rng.choice(DEPARTMENTS)
        years = rng.randint(0, 18)
        training = max(0, min(80, int(rng.gauss(28, 15))))
        projects = rng.randint(1, 12)
        attendance = round(max(70, min(100, rng.gauss(94, 5))), 1)
        manager_rating = rng.choices([1, 2, 3, 4, 5], weights=[3, 9, 27, 41, 20])[0]
        peer_score = rng.choices([1, 2, 3, 4, 5], weights=[4, 10, 28, 38, 20])[0]
        monthly_hours = rng.randint(120, 220)
        promoted = rng.choices(["Yes", "No"], weights=[18, 82])[0]

        latent_score = (
            manager_rating * 0.25
            + peer_score * 0.18
            + attendance / 100 * 1.35
            + training / 80 * 0.35
            + projects / 12 * 0.60
            + (0.35 if promoted == "Yes" else 0)
            + years / 18 * 0.15
            + rng.gauss(0, 0.28)
        )
        if latent_score < 3.25:
            performance = "Needs Improvement"
        elif latent_score < 4.12:
            performance = "Meets Expectations"
        else:
            performance = "Exceeds Expectations"

        dataset.append(
            {
                "EmployeeID": f"SYN{index:04d}",
                "Department": department,
                "JobRole": rng.choice(ROLES[department]),
                "EducationLevel": rng.choice(["High School", "Bachelor's", "Master's", "Doctorate"]),
                "YearsAtCompany": years,
                "TrainingHours": training,
                "ProjectsCompleted": projects,
                "AttendanceRate": attendance,
                "ManagerRating": manager_rating,
                "PeerFeedbackScore": peer_score,
                "MonthlyHours": monthly_hours,
                "PromotionLast2Years": promoted,
                "PerformanceLevel": performance,
            }
        )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--unlabeled", action="store_true", help="omit the target column for prediction examples"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/employee_performance_synthetic.csv")
    )
    args = parser.parse_args()
    if args.rows < 30 and not args.unlabeled:
        parser.error("--rows must be at least 30")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.rows, args.seed)
    fields = [field for field in FIELDS if not (args.unlabeled and field == "PerformanceLevel")]
    if args.unlabeled:
        for row in rows:
            row.pop("PerformanceLevel")
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {args.rows} synthetic rows: {args.output}")


if __name__ == "__main__":
    main()
