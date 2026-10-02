"""Clean logs: fix types, add timestamps, remove duplicates and implausible sensor values."""

import pandas as pd

from analyzer.config import KEY_COLUMNS, VALID_RANGES, VED_EPOCH


def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add an absolute `timestamp` column.

    In VED, `day_num` is the trip start (1.0 = 2017-11-01 00:00) and `time_ms`
    is the time elapsed since the trip started.
    """
    df = df.copy()
    trip_start = VED_EPOCH + pd.to_timedelta(df["day_num"] - 1, unit="D")
    df["timestamp"] = (trip_start + pd.to_timedelta(df["time_ms"], unit="ms")).dt.round("ms")
    return df


def remove_invalid_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace values outside their plausible range with NaN.

    Returns the cleaned frame and the number of values removed per column.
    """
    df = df.copy()
    removed = {}
    for column, (low, high) in VALID_RANGES.items():
        if column not in df:
            continue
        invalid = (df[column] < low) | (df[column] > high)
        if invalid.any():
            removed[column] = int(invalid.sum())
            df.loc[invalid, column] = float("nan")
    return df, removed


def clean_ved(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw VED logs into an analysis-ready frame.

    Steps: drop rows without keys, drop duplicate readings, sort each trip by time,
    remove implausible values, drop rows with no speed and no RPM, add timestamps.
    """
    df = df.dropna(subset=KEY_COLUMNS)
    df = df.astype({"vehicle_id": "int32", "trip_id": "int32", "time_ms": "int64"})
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    df = df.sort_values(KEY_COLUMNS)
    df, _ = remove_invalid_values(df)
    df = df.dropna(subset=["speed_kmh", "engine_rpm"], how="all")
    df = add_timestamp(df)
    return df.reset_index(drop=True)


def add_vehicle_info(df: pd.DataFrame, vehicle_info: pd.DataFrame) -> pd.DataFrame:
    """Attach vehicle type and class to every row."""
    columns = ["vehicle_id", "vehicle_type", "vehicle_class"]
    return df.merge(vehicle_info[columns], on="vehicle_id", how="left")


def summarize(df: pd.DataFrame) -> dict:
    """Quick summary of a cleaned log frame."""
    return {
        "rows": len(df),
        "vehicles": df["vehicle_id"].nunique(),
        "trips": df.groupby(["vehicle_id", "trip_id"]).ngroups,
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
    }


def clean_trip_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a user-supplied log (same steps as clean_ved, without VED's DayNum timestamps)."""
    df = df.dropna(subset=KEY_COLUMNS)
    df = df.astype({"vehicle_id": "int32", "trip_id": "int32", "time_ms": "int64"})
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    df = df.sort_values(KEY_COLUMNS)
    df, _ = remove_invalid_values(df)
    df = df.dropna(subset=["speed_kmh", "engine_rpm"], how="all")
    return df.reset_index(drop=True)
