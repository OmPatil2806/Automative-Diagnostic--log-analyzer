"""Tests for log loading and cleaning."""

import pandas as pd
import pytest

from analyzer.config import VED_COLUMNS
from analyzer.ingestion.cleaner import (
    add_timestamp,
    add_vehicle_info,
    clean_ved,
    remove_invalid_values,
    summarize,
)
from analyzer.ingestion.loader import list_ved_files, load_ved, load_vehicle_info


# --- loader ---------------------------------------------------------------

def test_list_ved_files_sorted(ved_dir):
    names = [p.name for p in list_ved_files(ved_dir)]
    assert names == ["VED_171101_week.csv", "VED_171108_week.csv"]


def test_load_ved_renames_columns(ved_dir):
    df = load_ved(ved_dir)
    assert list(df.columns) == list(VED_COLUMNS.values())
    assert len(df) == 7


def test_load_ved_limits_weeks(ved_dir):
    assert len(load_ved(ved_dir, weeks=1)) == 5


def test_load_ved_filters_vehicles(ved_dir):
    df = load_ved(ved_dir, vehicle_ids=[10])
    assert set(df["vehicle_id"]) == {10}


def test_load_ved_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_ved.py"):
        load_ved(tmp_path)


def test_load_ved_missing_columns_raises(tmp_path):
    pd.DataFrame({"VehId": [1]}).to_csv(tmp_path / "VED_171101_week.csv", index=False)
    with pytest.raises(ValueError, match="missing VED columns"):
        load_ved(tmp_path)


def test_load_vehicle_info_merges_both_files(static_dir):
    info = load_vehicle_info(static_dir)
    assert sorted(info["vehicle_id"]) == [8, 9, 10, 11]
    assert dict(zip(info["vehicle_id"], info["vehicle_type"])) == {
        8: "ICE", 9: "HEV", 10: "EV", 11: "PHEV",
    }
    assert info["engine"].isna().sum() == 2       # "NO DATA" -> missing
    assert info["weight_lb"].dtype == "float64"


# --- cleaner --------------------------------------------------------------

def test_add_timestamp():
    df = pd.DataFrame({"day_num": [1.0, 1.5], "time_ms": [0, 90_000]})
    ts = add_timestamp(df)["timestamp"]
    assert ts[0] == pd.Timestamp("2017-11-01 00:00:00")
    assert ts[1] == pd.Timestamp("2017-11-01 12:01:30")


def test_remove_invalid_values():
    df = pd.DataFrame({"speed_kmh": [50.0, -5.0, 300.0], "engine_rpm": [900.0, 1000.0, 1100.0]})
    cleaned, removed = remove_invalid_values(df)
    assert cleaned["speed_kmh"].isna().tolist() == [False, True, True]
    assert removed == {"speed_kmh": 2}


def test_clean_ved(ved_dir):
    df = clean_ved(load_ved(ved_dir))
    # duplicate and the row with no speed/RPM are removed
    assert len(df) == 5
    # trip 706 is sorted by time
    assert df.loc[df["trip_id"] == 706, "time_ms"].tolist() == [0, 1000, 2000]
    # impossible speed becomes NaN but the row (with valid RPM) stays
    assert df["speed_kmh"].isna().sum() == 1
    assert df["timestamp"].notna().all()


def test_add_vehicle_info(ved_dir, static_dir):
    df = add_vehicle_info(clean_ved(load_ved(ved_dir)), load_vehicle_info(static_dir))
    types = df.groupby("vehicle_id")["vehicle_type"].first().to_dict()
    assert types == {8: "ICE", 10: "EV"}


def test_summarize(ved_dir):
    stats = summarize(clean_ved(load_ved(ved_dir)))
    assert stats["vehicles"] == 2
    assert stats["trips"] == 3
