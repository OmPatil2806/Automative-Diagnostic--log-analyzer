"""Compute a 0-100 vehicle health score from faults and anomalies.

    score = 100 - DTC penalties - anomaly penalties

- Each DTC costs points by severity (critical 50, high 35, medium 25, low 10),
  so any medium or worse DTC moves the vehicle out of "good".
- Each anomaly episode NOT explained by a DTC (an early warning) costs
  10-25 points depending on how strong it is, scaled down if it is shorter
  than a minute, so a strong, persistent anomaly moves the vehicle out of
  "good" even before any code is set. Anomalies linked to a DTC are already
  covered by that DTC's penalty. Anomaly penalties are capped at 40 in total.

Status: 80-100 good, 50-79 needs attention, 0-49 critical. Any critical DTC
makes the status critical whatever the score.
"""

import pandas as pd

DTC_PENALTY = {"critical": 50, "high": 35, "medium": 25, "low": 10, "unknown": 25}
MAX_ANOMALY_PENALTY = 40

GOOD, ATTENTION, CRITICAL = "good", "needs attention", "critical"


def anomaly_penalty(peak_score: float, duration_s: float) -> float:
    """10 points for a borderline anomaly up to 25 for a strong one, less if short (< 60 s)."""
    strength = min(max(peak_score - 1, 0), 3)
    return (10 + 5 * strength) * min(1.0, duration_s / 60)


def trip_health(dtc_severities: list[str], unexplained_anomalies: list[tuple[float, float]]) -> int:
    """Health score for one trip.

    Args:
        dtc_severities: severity of each DTC set in the trip.
        unexplained_anomalies: (peak_score, duration_s) of episodes not linked to a DTC.
    """
    dtc = sum(DTC_PENALTY.get(s, DTC_PENALTY["unknown"]) for s in dtc_severities)
    anomalies = min(MAX_ANOMALY_PENALTY, sum(anomaly_penalty(p, d) for p, d in unexplained_anomalies))
    return int(round(max(0.0, 100 - dtc - anomalies)))


def health_status(score: int, dtc_severities: list[str] = ()) -> str:
    if score < 50 or "critical" in dtc_severities:
        return CRITICAL
    if score < 80:
        return ATTENTION
    return GOOD


def vehicle_health(trips: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle health from per-trip results (needs vehicle_id, health_score, status).

    A vehicle is only as healthy as its worst recent trip.
    """
    order = {GOOD: 0, ATTENTION: 1, CRITICAL: 2}
    rows = []
    for vehicle_id, df in trips.groupby("vehicle_id"):
        worst = df.loc[df["health_score"].idxmin()]
        rows.append({
            "vehicle_id": vehicle_id,
            "trips": len(df),
            "health_score": int(df["health_score"].min()),
            "status": max(df["status"], key=order.get),
            "trips_with_issues": int((df["status"] != GOOD).sum()),
            "worst_trip_id": worst["trip_id"],
        })
    return pd.DataFrame(rows).sort_values("health_score").reset_index(drop=True)
