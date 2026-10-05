"""Load VED OBD-II CSV logs and vehicle info into pandas DataFrames."""

from pathlib import Path

import pandas as pd

from analyzer.config import VED_COLUMNS, VED_DYNAMIC_DIR, VED_STATIC_DIR


def list_ved_files(directory: Path = VED_DYNAMIC_DIR) -> list[Path]:
    """Return the weekly VED CSV files in chronological order."""
    return sorted(Path(directory).glob("VED_*_week.csv"))


def load_ved_file(path: Path) -> pd.DataFrame:
    """Load one weekly VED CSV and rename its columns to the standard names."""
    df = pd.read_csv(path, low_memory=False)
    missing = set(VED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing VED columns: {sorted(missing)}")
    return df[list(VED_COLUMNS)].rename(columns=VED_COLUMNS)


def load_ved(
    directory: Path = VED_DYNAMIC_DIR,
    weeks: int | None = None,
    vehicle_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Load VED driving logs.

    Args:
        directory: folder containing the weekly VED CSV files.
        weeks: load only the first N weeks (None = all).
        vehicle_ids: keep only these vehicles (None = all).
    """
    files = list_ved_files(directory)
    if not files:
        raise FileNotFoundError(
            f"No VED files found in {directory}. Run `python scripts/download_ved.py` first."
        )
    frames = []
    for path in files[:weeks]:
        df = load_ved_file(path)
        if vehicle_ids is not None:
            df = df[df["vehicle_id"].isin(vehicle_ids)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_vehicle_info(directory: Path = VED_STATIC_DIR) -> pd.DataFrame:
    """Load the static vehicle info (type, class, engine, weight) for all VED vehicles."""
    files = sorted(Path(directory).glob("VED_Static_Data_*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No VED static files found in {directory}. Run `python scripts/download_ved.py` first."
        )
    frames = []
    for path in files:
        df = pd.read_excel(path)
        # The two files name the vehicle type column differently.
        df = df.rename(columns={"Vehicle Type": "vehicle_type", "EngineType": "vehicle_type"})
        frames.append(df)
    info = pd.concat(frames, ignore_index=True).rename(
        columns={
            "VehId": "vehicle_id",
            "Vehicle Class": "vehicle_class",
            "Engine Configuration & Displacement": "engine",
            "Transmission": "transmission",
            "Drive Wheels": "drive_wheels",
            "Generalized_Weight": "weight_lb",
        }
    )
    info = info.replace("NO DATA", pd.NA)
    info["weight_lb"] = pd.to_numeric(info["weight_lb"], errors="coerce")
    return info.drop_duplicates("vehicle_id").reset_index(drop=True)


# --- user-supplied trip logs ----------------------------------------------

REQUIRED_LOG_COLUMNS = ["time_ms", "speed_kmh", "engine_rpm"]
OPTIONAL_LOG_COLUMNS = [
    "maf_gs", "absolute_load_pct",
    "stft_b1_pct", "ltft_b1_pct", "stft_b2_pct", "ltft_b2_pct",
    "hv_battery_voltage_v", "hv_battery_current_a", "hv_battery_soc_pct",
]


def load_trip_log(path: Path, vehicle_id: int | None = None) -> pd.DataFrame:
    """Load a driving log CSV supplied by the user.

    Accepts the project's standard column names or raw VED names. Requires
    time_ms, speed_kmh and engine_rpm; other signals are optional and added as
    empty columns if missing. `vehicle_id` (argument, else a column, else -1
    for "unknown vehicle") selects the learned per-vehicle baseline.
    `trip_id` defaults to 1, so a file may also hold several trips.
    """
    df = pd.read_csv(path).rename(columns=VED_COLUMNS)
    missing = [c for c in REQUIRED_LOG_COLUMNS if c not in df]
    if missing:
        raise ValueError(
            f"{Path(path).name} is missing required columns: {missing}. "
            f"Required: {REQUIRED_LOG_COLUMNS}; optional: {OPTIONAL_LOG_COLUMNS}"
        )
    if vehicle_id is not None:
        df["vehicle_id"] = vehicle_id
    elif "vehicle_id" not in df:
        df["vehicle_id"] = -1
    else:
        df["vehicle_id"] = df["vehicle_id"].fillna(-1)   # empty cells = unknown vehicle, not a bad row
    df["trip_id"] = df["trip_id"].fillna(1) if "trip_id" in df else 1
    for column in OPTIONAL_LOG_COLUMNS:
        if column not in df:
            df[column] = float("nan")
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
