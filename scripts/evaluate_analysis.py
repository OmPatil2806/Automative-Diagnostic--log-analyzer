"""Diagnose every synthetic trip and check the results against the ground truth.

Usage:
    python scripts/evaluate_analysis.py

Requires a trained model (scripts/train_model.py) and synthetic data.

Outputs:
    reports/diagnosis_trips.csv      one row per trip: health score, status, likely cause
    reports/diagnosis_vehicles.csv   one row per vehicle
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.analysis.diagnosis import diagnose, to_frame  # noqa: E402
from analyzer.analysis.health_score import GOOD, vehicle_health  # noqa: E402
from analyzer.config import MODELS_DIR, REPORTS_DIR, SYNTHETIC_DIR  # noqa: E402
from analyzer.detection.anomaly import AnomalyDetector  # noqa: E402
from analyzer.synthetic.generator import NO_FAULT  # noqa: E402


def main() -> None:
    model_path = MODELS_DIR / "anomaly_detector.joblib"
    if not model_path.exists():
        sys.exit(f"{model_path} not found. Run `python scripts/train_model.py` first.")

    detector = AnomalyDetector.load(model_path)
    logs = pd.read_parquet(SYNTHETIC_DIR / "synthetic_logs.parquet")
    events = pd.read_csv(SYNTHETIC_DIR / "synthetic_dtc_events.csv")
    labels = pd.read_csv(SYNTHETIC_DIR / "synthetic_labels.csv")

    print(f"Diagnosing {len(labels)} trips ...")
    trips = to_frame(diagnose(detector.score(logs), events))
    trips = trips.merge(labels[["vehicle_id", "trip_id", "fault"]], on=["vehicle_id", "trip_id"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    trips.to_csv(REPORTS_DIR / "diagnosis_trips.csv", index=False)
    vehicle_health(trips).to_csv(REPORTS_DIR / "diagnosis_vehicles.csv", index=False)

    faulty = trips[trips["fault"] != NO_FAULT]
    normal = trips[trips["fault"] == NO_FAULT]
    with_cause = faulty[faulty["likely_cause"].notna()]
    correct = with_cause["likely_cause"] == with_cause["fault"]
    # "lean, cause unclear" is a correct but incomplete answer for the two lean faults
    unclear = (with_cause["likely_cause"] == "lean_unclear") & with_cause["fault"].isin(["vacuum_leak", "maf_drift"])

    print("\nRoot cause (from signals only, not from the DTC):")
    print(f"  faulty trips with an anomaly: {len(with_cause)}/{len(faulty)}")
    print(f"  correct cause:            {correct.mean():.0%}")
    print(f"  lean, cause unclear:      {unclear.mean():.0%}")
    print(f"  wrong:                    {(~correct & ~unclear).mean():.0%}")
    per_fault = with_cause.assign(correct=correct, unclear=unclear).groupby("fault")[["correct", "unclear"]].mean()
    print("  per fault (correct / unclear):")
    for fault, row in per_fault.iterrows():
        print(f"    {fault:24s} {row['correct']:.0%} / {row['unclear']:.0%}")

    print("\nHealth score:")
    print(f"  normal trips: mean {normal['health_score'].mean():.0f}, "
          f"{(normal['status'] == GOOD).mean():.0%} rated good")
    print(f"  faulty trips: mean {faulty['health_score'].mean():.0f}, "
          f"{(faulty['status'] != GOOD).mean():.0%} flagged (needs attention or critical)")
    print(f"\nSaved to {REPORTS_DIR.name}/diagnosis_trips.csv and diagnosis_vehicles.csv")


if __name__ == "__main__":
    main()
