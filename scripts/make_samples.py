"""Create small example input files in data/samples/ from the synthetic dataset.

Usage:
    python scripts/make_samples.py

For each fault type (and one normal trip) it picks a trip that the trained
model diagnoses correctly and writes:
    data/samples/<name>.csv       the driving log (input for run_pipeline.py)
    data/samples/<name>_dtc.csv   the fault codes with the time they were set
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.analysis.diagnosis import diagnose, to_frame  # noqa: E402
from analyzer.config import DATA_DIR, MODELS_DIR, SYNTHETIC_DIR  # noqa: E402
from analyzer.detection.anomaly import AnomalyDetector  # noqa: E402
from analyzer.ingestion.loader import OPTIONAL_LOG_COLUMNS, REQUIRED_LOG_COLUMNS  # noqa: E402
from analyzer.synthetic.generator import NO_FAULT  # noqa: E402

SAMPLES_DIR = DATA_DIR / "samples"
COLUMNS = ["vehicle_id", "trip_id", "timestamp"] + REQUIRED_LOG_COLUMNS + OPTIONAL_LOG_COLUMNS
MIN_ROWS, MAX_ROWS = 400, 1200


def main() -> None:
    detector = AnomalyDetector.load(MODELS_DIR / "anomaly_detector.joblib")
    logs = pd.read_parquet(SYNTHETIC_DIR / "synthetic_logs.parquet")
    events = pd.read_csv(SYNTHETIC_DIR / "synthetic_dtc_events.csv")
    labels = pd.read_csv(SYNTHETIC_DIR / "synthetic_labels.csv")

    sizes = logs.groupby(["vehicle_id", "trip_id"]).size().rename("rows").reset_index()
    trips = to_frame(diagnose(detector.score(logs), events)).merge(labels, on=["vehicle_id", "trip_id"])
    trips = trips.merge(sizes, on=["vehicle_id", "trip_id"])
    trips = trips[trips["rows"].between(MIN_ROWS, MAX_ROWS)]
    good = trips[(trips["fault"] == NO_FAULT) & (trips["status"] == "good") & (trips["anomaly_episodes"] == 0)]
    correct = trips[(trips["fault"] != NO_FAULT) & (trips["likely_cause"] == trips["fault"])]

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    picks = [("normal_trip", good)] + [(f, df) for f, df in correct.groupby("fault")]
    for name, candidates in picks:
        if candidates.empty:
            print(f"  skipped {name}: no suitable trip")
            continue
        pick = candidates.sort_values("rows").iloc[len(candidates) // 2]
        key = (pick.vehicle_id, pick.trip_id)
        trip = logs[(logs["vehicle_id"] == key[0]) & (logs["trip_id"] == key[1])][COLUMNS]
        numeric = trip.select_dtypes("number").columns
        trip = trip.assign(**trip[numeric].round(3))
        trip.to_csv(SAMPLES_DIR / f"{name}.csv", index=False)
        codes = events[(events["vehicle_id"] == key[0]) & (events["trip_id"] == key[1])][["code", "time_ms"]]
        if len(codes):
            codes.to_csv(SAMPLES_DIR / f"{name}_dtc.csv", index=False)
        print(f"  {name:24s} vehicle {key[0]}, trip {key[1]}, {len(trip)} rows, DTC: {','.join(codes['code']) or '-'}")
    print(f"Saved to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
