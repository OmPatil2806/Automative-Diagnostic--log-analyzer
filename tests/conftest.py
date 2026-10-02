"""Shared pytest fixtures (small sample logs)."""

import numpy as np
import pandas as pd
import pytest

from analyzer.config import VED_COLUMNS


@pytest.fixture
def raw_ved_frame() -> pd.DataFrame:
    """A tiny log in raw VED format: 2 vehicles, 3 trips, with a few bad rows."""
    rows = [
        # DayNum, VehId, Trip, Timestamp(ms), speed, rpm
        (1.5, 8, 706, 1000, 45.0, 2300.0),
        (1.5, 8, 706, 0, 40.0, 2285.0),       # out of order
        (1.5, 8, 706, 0, 40.0, 2285.0),       # duplicate
        (1.5, 8, 706, 2000, 300.0, 2400.0),   # impossible speed
        (2.0, 8, 707, 0, 0.0, 800.0),
        (3.25, 10, 15, 0, 20.0, None),
        (3.25, 10, 15, 500, None, None),      # no speed and no RPM
    ]
    df = pd.DataFrame(columns=list(VED_COLUMNS), dtype="float64")
    for i, (day, veh, trip, ts, speed, rpm) in enumerate(rows):
        df.loc[i] = float("nan")
        df.loc[i, ["DayNum", "VehId", "Trip", "Timestamp(ms)"]] = [day, veh, trip, ts]
        df.loc[i, ["Vehicle Speed[km/h]", "Engine RPM[RPM]"]] = [speed, rpm]
        df.loc[i, ["Latitude[deg]", "Longitude[deg]"]] = [42.27, -83.73]
    return df


@pytest.fixture
def ved_dir(tmp_path, raw_ved_frame):
    """A folder with two weekly VED CSV files."""
    raw_ved_frame.iloc[:5].to_csv(tmp_path / "VED_171101_week.csv", index=False)
    raw_ved_frame.iloc[5:].to_csv(tmp_path / "VED_171108_week.csv", index=False)
    return tmp_path


@pytest.fixture
def static_dir(tmp_path):
    """A folder with the two VED static files (they use different column names)."""
    common = {
        "Vehicle Class": ["Car", "SUV"],
        "Engine Configuration & Displacement": ["I4 2.0L", "NO DATA"],
        "Transmission": ["NO DATA", "CVT"],
        "Drive Wheels": ["FWD", "NO DATA"],
        "Generalized_Weight": [3500, "NO DATA"],
    }
    pd.DataFrame({"VehId": [8, 9], "Vehicle Type": ["ICE", "HEV"], **common}).to_excel(
        tmp_path / "VED_Static_Data_ICE&HEV.xlsx", index=False
    )
    pd.DataFrame({"VehId": [10, 11], "EngineType": ["EV", "PHEV"], **common}).to_excel(
        tmp_path / "VED_Static_Data_PHEV&EV.xlsx", index=False
    )
    return tmp_path


def make_trip(vehicle_id: int, trip_id: int, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A realistic-looking clean trip with every signal the faults need."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1000
    # whole km/h, like real OBD-II speed readings
    speed = np.clip(np.round(40 + 20 * np.sin(np.arange(n) / 20) + rng.normal(0, 2, n)), 0, None)
    return pd.DataFrame({
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "time_ms": t,
        "timestamp": pd.Timestamp("2017-11-01") + pd.to_timedelta(t, unit="ms"),
        "speed_kmh": speed,
        "engine_rpm": 1500 + speed * 20 + rng.normal(0, 30, n),
        "maf_gs": 8 + speed * 0.2,
        "absolute_load_pct": 30 + rng.normal(0, 5, n),
        "stft_b1_pct": rng.normal(0, 2, n),
        "ltft_b1_pct": rng.normal(1, 0.5, n),
        "stft_b2_pct": rng.normal(0, 2, n),
        "ltft_b2_pct": rng.normal(1, 0.5, n),
        "hv_battery_current_a": rng.normal(-15, 30, n),
        "hv_battery_voltage_v": 300 + rng.normal(0, 3, n),
        "hv_battery_soc_pct": np.linspace(60, 50, n),
    })
