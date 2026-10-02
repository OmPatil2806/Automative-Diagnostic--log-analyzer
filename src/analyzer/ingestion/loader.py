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
