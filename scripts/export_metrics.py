"""Evaluate the trained model on the synthetic trips and save the results for the dashboard.

Usage:
    python scripts/export_metrics.py

Needs the trained model (models/anomaly_detector.joblib) and the synthetic data
(scripts/prepare_ved.py --weeks 1, then scripts/generate_synthetic.py).
Writes dashboard/assets/model_metrics.json (small, committed), which the
Model performance page reads, so it works without the raw data.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analyzer.analysis.diagnosis import diagnose, to_frame  # noqa: E402
from analyzer.config import MODELS_DIR, PROCESSED_DIR, SYNTHETIC_DIR  # noqa: E402
from analyzer.detection.anomaly import AnomalyDetector  # noqa: E402
from analyzer.detection.evaluate import row_metrics, summarize, trip_results  # noqa: E402
from analyzer.synthetic.generator import NO_FAULT  # noqa: E402

OUT = ROOT / "dashboard" / "assets" / "model_metrics.json"
FAULTS = ["vacuum_leak", "maf_drift", "rich_injector", "misfire", "speed_sensor_failure", "hv_battery_degradation"]
PERSISTENCE_GRID = [(10, 0.5), (20, 0.6), (30, 0.7), (40, 0.8), (60, 0.8), (90, 0.9)]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else round(float(x), 3)


def main() -> None:
    model_path = MODELS_DIR / "anomaly_detector.joblib"
    for path, hint in [(model_path, "train_model.py"), (SYNTHETIC_DIR / "synthetic_logs.parquet", "generate_synthetic.py")]:
        if not path.exists():
            sys.exit(f"{path} not found. Run `python scripts/{hint}` first.")

    detector = AnomalyDetector.load(model_path)
    logs = pd.read_parquet(SYNTHETIC_DIR / "synthetic_logs.parquet")
    events = pd.read_csv(SYNTHETIC_DIR / "synthetic_dtc_events.csv")
    labels = pd.read_csv(SYNTHETIC_DIR / "synthetic_labels.csv")
    labels[["onset_time_ms", "dtc_time_ms"]] = labels[["onset_time_ms", "dtc_time_ms"]].astype("Int64")

    print("Scoring the synthetic trips three ways ...")
    methods = {}
    for method in ("forest", "range", "combined"):
        scored = detector.score(logs, method=method)
        rows = row_metrics(scored, logs)
        per_fault = summarize(trip_results(scored, labels)).set_index("fault")
        methods[method] = {
            "precision": _num(rows["precision"]), "recall": _num(rows["recall"]), "f1": _num(rows["f1"]),
            "false_alarm_rate": _num(per_fault.loc[NO_FAULT, "false_alarm_rate"]),
            "detection_rate": {f: _num(per_fault.loc[f, "detection_rate"]) for f in FAULTS},
        }
    scored = detector.score(logs)
    trips = trip_results(scored, labels)
    per_fault = summarize(trips).set_index("fault")

    print("Alarm rule trade-off ...")
    original = (detector.persistence_window, detector.persistence_ratio)
    grid = []
    for window, ratio in PERSISTENCE_GRID:
        detector.persistence_window, detector.persistence_ratio = window, ratio
        t = summarize(trip_results(detector.score(logs), labels)).set_index("fault")
        grid.append({"window": window, "ratio": ratio, "false_alarm_rate": _num(t.loc[NO_FAULT, "false_alarm_rate"]),
                     "detected": _num(t.drop(NO_FAULT)["detection_rate"].mean()),
                     "before_dtc": _num(t.drop(NO_FAULT)["detected_before_dtc"].mean()),
                     "used": (window, ratio) == original})
    detector.persistence_window, detector.persistence_ratio = original

    print("Root cause and health score ...")
    diag = to_frame(diagnose(scored, events)).merge(labels[["vehicle_id", "trip_id", "fault"]], on=["vehicle_id", "trip_id"])
    faulty = diag[diag["fault"] != NO_FAULT]
    with_cause = faulty[faulty["likely_cause"].notna()]
    correct = with_cause["likely_cause"] == with_cause["fault"]
    normal = diag[diag["fault"] == NO_FAULT]

    metrics = {
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "model": {"detectors": list(detector.models),
                  "alarm_rule": {"window": original[0], "ratio": original[1]},
                  "vehicles_with_baseline": len(set(detector.baseline.maf_k) | set(detector.baseline.stft_offset)
                                                | set(detector.baseline.battery_coef))},
        "dataset": {"trips": int(len(labels)), "normal_trips": int((labels["fault"] == NO_FAULT).sum()),
                    "faulty_trips": int((labels["fault"] != NO_FAULT).sum()), "readings": int(len(logs)),
                    "training_trips": int(pd.read_parquet(PROCESSED_DIR / "ved_clean.parquet", columns=["vehicle_id", "trip_id"])
                                          .drop_duplicates().shape[0] - len(labels))
                    if (PROCESSED_DIR / "ved_clean.parquet").exists() else None},
        "methods": methods,
        "per_fault": [{
            "fault": f, "trips": int(per_fault.loc[f, "trips"]),
            "detected": _num(per_fault.loc[f, "detection_rate"]),
            "before_dtc": _num(per_fault.loc[f, "detected_before_dtc"]),
            "median_warning_s": _num(per_fault.loc[f, "median_warning_s"]),
            "suspect_group": per_fault.loc[f, "top_suspect_group"],
            "root_cause_accuracy": _num(correct[with_cause["fault"] == f].mean()),
        } for f in FAULTS],
        "warning_times": {f: [round(float(w), 1) for w in trips.loc[(trips["fault"] == f) & trips["alarm_raised"],
                                                                     "warning_before_dtc_s"].dropna()] for f in FAULTS},
        "persistence": grid,
        "diagnosis": {"root_cause_accuracy": _num(correct.mean()),
                      "faulty_with_anomaly": int(len(with_cause)),
                      "normal_good_rate": _num((normal["status"] == "good").mean()),
                      "normal_mean_score": _num(normal["health_score"].mean()),
                      "faulty_flagged_rate": _num((faulty["status"] != "good").mean()),
                      "faulty_mean_score": _num(faulty["health_score"].mean())},
    }
    OUT.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    m = methods["combined"]
    print(f"F1 {m['f1']:.2f} | false alarms {m['false_alarm_rate']:.1%} | root cause "
          f"{metrics['diagnosis']['root_cause_accuracy']:.0%}")
    print(f"Saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
