"""Train the anomaly detector on normal VED trips and evaluate it on synthetic faults.

Usage:
    python scripts/train_model.py

Training uses only cleaned VED trips that are NOT in the synthetic dataset, so
the model never sees any evaluation trip (normal or faulty) during training.

Outputs:
    models/anomaly_detector.joblib
    reports/anomaly_trip_results.csv   one row per synthetic trip
    reports/anomaly_summary.csv        per-fault detection summary
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SYNTHETIC_DIR  # noqa: E402
from analyzer.detection.anomaly import AnomalyDetector  # noqa: E402
from analyzer.detection.evaluate import row_metrics, summarize, trip_results  # noqa: E402


def main() -> None:
    clean_path = PROCESSED_DIR / "ved_clean.parquet"
    synthetic_path = SYNTHETIC_DIR / "synthetic_logs.parquet"
    for path, step in [(clean_path, "prepare_ved.py"), (synthetic_path, "generate_synthetic.py")]:
        if not path.exists():
            sys.exit(f"{path} not found. Run `python scripts/{step}` first.")

    clean = pd.read_parquet(clean_path)
    logs = pd.read_parquet(synthetic_path)
    labels = pd.read_csv(SYNTHETIC_DIR / "synthetic_labels.csv")
    labels[["onset_time_ms", "dtc_time_ms"]] = labels[["onset_time_ms", "dtc_time_ms"]].astype("Int64")

    eval_trips = labels[["vehicle_id", "trip_id"]].assign(_eval=True)
    train = clean.merge(eval_trips, on=["vehicle_id", "trip_id"], how="left")
    train = train[train["_eval"].isna()].drop(columns="_eval")
    print(f"Training on {train.groupby(['vehicle_id', 'trip_id']).ngroups} normal trips ({len(train):,} rows) ...")

    detector = AnomalyDetector().fit(train)
    print(f"  trained detectors: {', '.join(detector.models)}")
    detector.save(MODELS_DIR / "anomaly_detector.joblib")

    print(f"Evaluating on {len(labels)} synthetic trips ...")
    scored = detector.score(logs)
    rows = row_metrics(scored, logs)
    trips = trip_results(scored, labels)
    summary = summarize(trips)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    trips.to_csv(REPORTS_DIR / "anomaly_trip_results.csv", index=False)
    summary.to_csv(REPORTS_DIR / "anomaly_summary.csv", index=False)

    print(f"\nRow level (fully developed faults vs normal): "
          f"precision {rows['precision']:.2f} | recall {rows['recall']:.2f} | F1 {rows['f1']:.2f}")
    print("\nTrip level:")
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 120):
        print(summary.to_string(index=False))
    print(f"\nSaved model to {MODELS_DIR.name}/ and results to {REPORTS_DIR.name}/")


if __name__ == "__main__":
    main()
