"""Data exploration (EDA) summary of the VED dataset.

The raw data is too large for the repository, so `build_eda_summary()`
reduces it to the numbers and chart data the Data exploration page needs,
and `scripts/export_eda_summary.py` saves that as a small JSON file in
dashboard/assets/ (committed), so the page works right after cloning.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analyzer.config import VALID_RANGES
from analyzer.ingestion.cleaner import remove_invalid_values

ASSETS = Path(__file__).resolve().parents[1] / "assets"
EDA_PATH = ASSETS / "eda_summary.json"
TRIP = ["vehicle_id", "trip_id"]

SIGNALS = {
    "speed_kmh": "Speed", "engine_rpm": "Engine RPM", "maf_gs": "Airflow (MAF)",
    "absolute_load_pct": "Engine load", "stft_b1_pct": "Fuel trim, short term (bank 1)",
    "ltft_b1_pct": "Fuel trim, long term (bank 1)", "stft_b2_pct": "Fuel trim (bank 2)",
    "outside_air_temp_c": "Outside air temperature", "hv_battery_voltage_v": "HV battery voltage",
    "hv_battery_current_a": "HV battery current", "hv_battery_soc_pct": "HV battery charge (SOC)",
    "fuel_rate_lph": "Fuel rate", "ac_power_kw": "A/C power", "heater_power_w": "Heater power",
}
DISTRIBUTIONS = {
    "speed_kmh": "Speed (km/h)",
    "engine_rpm": "Engine RPM (engine running)",
    "maf_gs": "Airflow, MAF (g/s)",
    "stft_b1_pct": "Short-term fuel trim (%)",
    "total_trim_b1_pct": "Total fuel trim, short + long (%)",
    "hv_battery_voltage_v": "HV battery voltage (V)",
}
TYPE_ORDER = ["ICE", "HEV", "PHEV", "EV"]
TYPE_NAMES = {"ICE": "Petrol (ICE)", "HEV": "Hybrid (HEV)", "PHEV": "Plug-in hybrid (PHEV)", "EV": "Electric (EV)"}

FINDINGS = [
    ("Each vehicle type records a different set of signals; no trip has all of them",
     "One detector per subsystem; the report lists skipped checks"),
    ("Fuel trim coverage jumped from about 15% to 70-90% of trips after the first week",
     "Train on several weeks; the included model (week 1 only) is conservative for fuel faults"),
    ("Usual fuel trim and battery voltage differ widely between vehicles",
     "Per-vehicle baselines; features measure deviation from them"),
    ("Airflow is about constant x RPM x load per engine; voltage is about linear in charge and current",
     "Physics-based residuals for the air and battery checks"),
    ("About one reading per second", "Fine for fuel trims and voltage; misfire detection is limited"),
    ("VED has no fault codes", "Faults are simulated on real trips to get labeled data"),
    ("Very few impossible values", "Simple range-based cleaning is enough"),
]


def _histogram(values: pd.Series, bins: int = 60) -> dict:
    """Compact histogram (central 99.8% of values) plus summary stats."""
    values = values.dropna()
    if values.empty:
        return {"edges": [], "counts": [], "median": None, "p1": None, "p99": None, "n": 0}
    lo, hi = values.quantile([0.001, 0.999])
    counts, edges = np.histogram(values.clip(lo, hi), bins=bins, range=(lo, hi))
    return {
        "edges": [round(float(e), 3) for e in edges],
        "counts": counts.tolist(),
        "median": round(float(values.median()), 2),
        "p1": round(float(values.quantile(0.01)), 2),
        "p99": round(float(values.quantile(0.99)), 2),
        "n": int(len(values)),
    }


def _sample_points(df: pd.DataFrame, columns: list[str], n: int, seed: int = 0) -> dict:
    points = df[columns].dropna()
    points = points.sample(min(n, len(points)), random_state=seed) if len(points) else points
    return {c: [round(float(v), 2) for v in points[c]] for c in columns}


def build_eda_summary(raw: pd.DataFrame, clean: pd.DataFrame, weeks: list[str]) -> dict:
    """Reduce raw + cleaned VED logs (clean must have vehicle_type and timestamp) to the page's data."""
    trips = clean.groupby(TRIP).agg(
        vehicle_type=("vehicle_type", "first"),
        minutes=("time_ms", lambda t: (t.max() - t.min()) / 60_000),
    )
    gaps = clean.groupby(TRIP)["time_ms"].diff().dropna()
    signals = [c for c in SIGNALS if c in clean]
    recorded = clean.groupby(TRIP)[signals].agg(lambda s: s.notna().mean() > 0.5)

    # vehicle types
    by_vehicle = clean.drop_duplicates("vehicle_id")["vehicle_type"].value_counts()
    by_reading = clean["vehicle_type"].value_counts(normalize=True) * 100

    # coverage per week (fuel trims and airflow)
    weekly = []
    for week in weeks:
        start = pd.Timestamp(datetime.strptime(week, "%y%m%d"))
        in_week = clean[(clean["timestamp"] >= start) & (clean["timestamp"] < start + pd.Timedelta(days=7))]
        if in_week.empty:
            continue
        rec = in_week.groupby(TRIP)[["stft_b1_pct", "maf_gs"]].agg(lambda s: s.notna().mean() > 0.5).mean() * 100
        weekly.append({"week": start.strftime("%b %d"), "fuel_trims": round(float(rec["stft_b1_pct"]), 1),
                       "airflow": round(float(rec["maf_gs"]), 1)})

    # distributions
    data = clean.assign(total_trim_b1_pct=clean["stft_b1_pct"] + clean["ltft_b1_pct"])
    distributions = {}
    for column, title in DISTRIBUTIONS.items():
        values = data[column]
        if column == "engine_rpm":
            values = values[values > 0]
        if values.notna().any():
            distributions[column] = {"title": title, **_histogram(values)}

    # per-vehicle differences
    per_vehicle = clean.groupby("vehicle_id").agg(ltft=("ltft_b1_pct", "median"),
                                                  hv=("hv_battery_voltage_v", "median"))

    # signal relationships
    air = clean[(clean["engine_rpm"] > 500) & (clean["absolute_load_pct"] > 5) & clean["maf_gs"].notna()]
    air = air.assign(rpm_x_load=air["engine_rpm"] * air["absolute_load_pct"] / 100)
    airflow = []
    for vehicle_id in air["vehicle_id"].value_counts().index[:3]:
        v = air[air["vehicle_id"] == vehicle_id]
        airflow.append({"vehicle_id": int(vehicle_id), "slope": round(float((v["maf_gs"] / v["rpm_x_load"]).median()), 5),
                        **_sample_points(v, ["rpm_x_load", "maf_gs"], 600)})
    battery = {}
    bcols = ["hv_battery_voltage_v", "hv_battery_soc_pct", "hv_battery_current_a"]
    batt = clean.dropna(subset=bcols)
    if len(batt):
        vehicle_id = batt["vehicle_id"].value_counts().index[0]
        b = batt[batt["vehicle_id"] == vehicle_id]
        X = np.column_stack([np.ones(len(b)), b["hv_battery_soc_pct"], b["hv_battery_current_a"]])
        coef, *_ = np.linalg.lstsq(X, b["hv_battery_voltage_v"], rcond=None)
        residual = b["hv_battery_voltage_v"] - X @ coef
        battery = {"vehicle_id": int(vehicle_id), "coef": [round(float(c), 3) for c in coef],
                   "typical_error_v": round(float(residual.abs().median()), 2),
                   **_sample_points(b, ["hv_battery_current_a", "hv_battery_voltage_v"], 1500)}

    # data quality
    _, invalid = remove_invalid_values(raw)
    quality = [
        {"issue": "Duplicate readings", "count": int(raw.duplicated(["vehicle_id", "trip_id", "time_ms"]).sum()),
         "action": "Removed"},
        {"issue": "Rows with neither speed nor RPM",
         "count": int((raw["speed_kmh"].isna() & raw["engine_rpm"].isna()).sum()), "action": "Removed"},
        *[{"issue": f"Impossible {c} (outside {VALID_RANGES[c][0]} to {VALID_RANGES[c][1]})", "count": int(n),
           "action": "Set to missing"} for c, n in invalid.items()],
    ]

    return {
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "weeks": len(weeks),
        "overview": {
            "raw_readings": int(len(raw)), "clean_readings": int(len(clean)),
            "vehicles": int(clean["vehicle_id"].nunique()), "trips": int(len(trips)),
            "first_day": clean["timestamp"].min().strftime("%Y-%m-%d"),
            "last_day": clean["timestamp"].max().strftime("%Y-%m-%d"),
            "median_trip_min": round(float(trips["minutes"].median()), 1),
            "median_gap_ms": int(gaps.median()), "p90_gap_ms": int(gaps.quantile(0.9)),
        },
        "vehicle_types": [{"type": t, "name": TYPE_NAMES[t], "vehicles": int(by_vehicle.get(t, 0)),
                           "readings_pct": round(float(by_reading.get(t, 0)), 1)} for t in TYPE_ORDER],
        "trip_minutes": _histogram(trips["minutes"].clip(upper=60), bins=60),
        "coverage": [{"signal": c, "name": SIGNALS[c], "trips_pct": round(float(recorded[c].mean() * 100), 1)}
                     for c in signals],
        "coverage_by_type": {
            TYPE_NAMES[t]: {SIGNALS[c]: round(float(v * 100)) for c, v in row.items()}
            for t, row in recorded.join(trips["vehicle_type"]).groupby("vehicle_type").mean().iterrows()
            if t in TYPE_NAMES
        },
        "weekly_coverage": weekly,
        "distributions": distributions,
        "per_vehicle": {
            "ltft_median": [round(float(x), 2) for x in per_vehicle["ltft"].dropna()],
            "ltft_hist": _histogram(per_vehicle["ltft"], bins=40),
            "hv_voltage_median": [round(float(x), 1) for x in per_vehicle["hv"].dropna().sort_values()],
        },
        "airflow": airflow,
        "battery": battery,
        "quality": quality,
        "findings": [{"finding": f, "decision": d} for f, d in FINDINGS],
    }


def save_eda_summary(summary: dict, path: Path = EDA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    return path


@st.cache_data(show_spinner=False)
def load_eda_summary(path: str = str(EDA_PATH)) -> dict | None:
    """The saved summary, or None if it has not been generated."""
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
