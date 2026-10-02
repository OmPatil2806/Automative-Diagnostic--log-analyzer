"""Evaluate anomaly detection against the synthetic ground truth."""

import numpy as np
import pandas as pd

from analyzer.synthetic.generator import NO_FAULT

TRIP_KEYS = ["vehicle_id", "trip_id"]


def row_metrics(scored: pd.DataFrame, logs: pd.DataFrame) -> dict[str, float]:
    """Precision / recall of `alarm` on rows: fully developed faults vs normal trips.

    Rows in the ramp-up phase are left out, since a half-developed fault is
    neither clearly normal nor clearly faulty.
    """
    df = scored.merge(logs[TRIP_KEYS + ["time_ms", "fault", "fault_intensity"]], on=TRIP_KEYS + ["time_ms"])
    df = df[(df["fault"] == NO_FAULT) | (df["fault_intensity"] == 1)]
    truth = df["fault"] != NO_FAULT
    pred = df["alarm"]
    tp, fp, fn = (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def trip_results(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per trip: was the fault detected, when, and how long before its DTC?

    - faulty trip: detected if an alarm is raised at or after fault onset
    - normal trip: false alarm if any alarm is raised
    """
    alarms = scored[scored["alarm"]].merge(labels[TRIP_KEYS + ["onset_time_ms"]], on=TRIP_KEYS)
    after_onset = alarms[alarms["onset_time_ms"].isna() | (alarms["time_ms"] >= alarms["onset_time_ms"])]
    first = after_onset.groupby(TRIP_KEYS).agg(
        first_alarm_ms=("time_ms", "min"),
    ).reset_index()
    suspect = (
        after_onset.sort_values("time_ms").groupby(TRIP_KEYS)["suspect_group"].agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None
        ).rename("suspect_group").reset_index()
    )
    out = labels.merge(first, on=TRIP_KEYS, how="left").merge(suspect, on=TRIP_KEYS, how="left")
    out["alarm_raised"] = out["first_alarm_ms"].notna()
    out["warning_before_dtc_s"] = (
        (out["dtc_time_ms"].astype("float64") - out["first_alarm_ms"]) / 1000
    )
    # share of ALL faulty trips caught before their DTC (undetected trips count as no)
    out["detected_before_dtc"] = (out["warning_before_dtc_s"] > 0).fillna(False).astype(bool)
    return out


def summarize(trips: pd.DataFrame) -> pd.DataFrame:
    """Per-fault summary: detection rate, share detected before the DTC, median lead time."""
    rows = []
    for fault, df in trips.groupby("fault"):
        if fault == NO_FAULT:
            rows.append({"fault": fault, "trips": len(df), "false_alarm_rate": df["alarm_raised"].mean()})
            continue
        detected = df[df["alarm_raised"]]
        rows.append({
            "fault": fault,
            "trips": len(df),
            "detection_rate": df["alarm_raised"].mean(),
            "detected_before_dtc": df["detected_before_dtc"].mean(),
            "median_warning_s": detected["warning_before_dtc_s"].median() if len(detected) else np.nan,
            "top_suspect_group": detected["suspect_group"].mode().iat[0] if detected["suspect_group"].notna().any() else None,
        })
    return pd.DataFrame(rows)
