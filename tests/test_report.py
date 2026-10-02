"""Tests for user log loading, DTC inputs, the pipeline and the reports."""

import json

import numpy as np
import pandas as pd
import pytest

from analyzer.detection.anomaly import AnomalyDetector
from analyzer.ingestion.cleaner import clean_trip_log
from analyzer.ingestion.loader import load_trip_log
from analyzer.pipeline import collect_dtcs, run
from analyzer.reporting.report import fmt_time
from analyzer.synthetic.faults import FAULTS
from analyzer.synthetic.generator import inject_fault
from tests.conftest import make_trip

LOG_COLUMNS = ["time_ms", "speed_kmh", "engine_rpm", "maf_gs", "absolute_load_pct",
               "stft_b1_pct", "ltft_b1_pct", "hv_battery_voltage_v", "hv_battery_current_a", "hv_battery_soc_pct"]


@pytest.fixture(scope="module")
def detector() -> AnomalyDetector:
    normal = pd.concat([make_trip(1, t, n=300, seed=t) for t in range(30)], ignore_index=True)
    return AnomalyDetector(n_estimators=50).fit(normal)


@pytest.fixture
def rich_log(tmp_path):
    """A log file (vehicle 1) with a rich injector fault, plus its DTC file."""
    trip, label = inject_fault(make_trip(1, 7, n=300, seed=7), FAULTS["rich_injector"], np.random.default_rng(0))
    log = tmp_path / "trip.csv"
    trip[["vehicle_id"] + LOG_COLUMNS].to_csv(log, index=False)
    dtc = tmp_path / "trip_dtc.csv"
    pd.DataFrame({"code": ["P0172"], "time_ms": [label["dtc_time_ms"]]}).to_csv(dtc, index=False)
    return log, dtc, label


# --- loading user logs ----------------------------------------------------

def test_load_trip_log_minimal_columns(tmp_path):
    path = tmp_path / "min.csv"
    pd.DataFrame({"time_ms": [0, 1000], "speed_kmh": [10, 12], "engine_rpm": [900, 950]}).to_csv(path, index=False)
    df = load_trip_log(path)
    assert (df["vehicle_id"] == -1).all() and (df["trip_id"] == 1).all()
    assert df["stft_b1_pct"].isna().all()          # optional signals added as empty


def test_load_trip_log_accepts_ved_names_and_vehicle_override(tmp_path):
    path = tmp_path / "ved.csv"
    pd.DataFrame({"Timestamp(ms)": [0, 1000], "Vehicle Speed[km/h]": [10, 12],
                  "Engine RPM[RPM]": [900, 950], "VehId": [8, 8]}).to_csv(path, index=False)
    df = load_trip_log(path, vehicle_id=42)
    assert {"time_ms", "speed_kmh", "engine_rpm"} <= set(df.columns)
    assert (df["vehicle_id"] == 42).all()


def test_load_trip_log_missing_required(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time_ms": [0], "speed_kmh": [10]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="engine_rpm"):
        load_trip_log(path)


def test_clean_trip_log_sorts_and_removes_invalid(tmp_path):
    path = tmp_path / "messy.csv"
    pd.DataFrame({"time_ms": [2000, 0, 1000, 1000], "speed_kmh": [10, 999, 12, 12],
                  "engine_rpm": [900, 950, 980, 980]}).to_csv(path, index=False)
    df = clean_trip_log(load_trip_log(path))
    assert df["time_ms"].tolist() == [0, 1000, 2000]
    assert np.isnan(df.loc[0, "speed_kmh"])


# --- DTC inputs -----------------------------------------------------------

def test_collect_dtcs_from_codes_and_obd_response():
    logs = make_trip(1, 1, n=50)
    events = collect_dtcs(logs, dtc_codes=["p0171"], obd_response="43 03 00 01 71")
    assert events["code"].tolist() == ["P0171", "P0300"]      # duplicate P0171 removed
    assert (events["time_ms"] == logs["time_ms"].max()).all()
    assert not events["time_known"].any()


def test_collect_dtcs_from_file(rich_log):
    log, dtc, label = rich_log
    events = collect_dtcs(clean_trip_log(load_trip_log(log)), dtc_file=dtc)
    assert events.iloc[0][["vehicle_id", "trip_id", "code", "time_ms"]].tolist() == [1, 1, "P0172", label["dtc_time_ms"]]
    assert events["time_known"].all()


def test_collect_dtcs_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid DTC"):
        collect_dtcs(make_trip(1, 1, n=10), dtc_codes=["XYZ"])


# --- pipeline and reports -------------------------------------------------

def test_pipeline_diagnoses_fault(rich_log, detector):
    log, dtc, _ = rich_log
    result = run(log, dtc_file=dtc, detector=detector)[0]
    r = result.report
    assert r["health"]["status"] == "needs attention"
    assert r["dtcs"][0]["code"] == "P0172" and r["dtcs"][0]["time_s"] is not None
    assert r["main_finding"]["cause"] == "rich_injector"
    assert r["main_finding"]["warning_before_dtc_s"] is not None
    assert r["recommended_checks"][0] == "Check fuel injectors for leaks"
    assert r["checks_run"]["Fuel system (fuel trims)"] == "checked"
    json.dumps(r, default=str)                      # JSON-ready

    assert "HEALTH SCORE:" in result.text and "P0172" in result.text and "Likely cause" in result.text
    assert result.html.startswith("<!doctype html>") and "P0172" in result.html and "plotly" in result.html


def test_pipeline_codes_without_time(rich_log, detector):
    log, _, _ = rich_log
    r = run(log, dtc_codes=["P0172"], detector=detector)[0].report
    assert r["dtcs"][0]["time_s"] is None
    assert r["main_finding"]["warning_before_dtc_s"] is None
    assert any("unknown" in n for n in r["notes"])


def test_pipeline_normal_trip_without_dtcs(tmp_path, detector):
    path = tmp_path / "normal.csv"
    make_trip(1, 3, n=300, seed=303)[["vehicle_id"] + LOG_COLUMNS].to_csv(path, index=False)
    result = run(path, detector=detector)[0]
    assert result.report["health"]["status"] == "good"
    assert result.report["recommended_checks"] == []
    assert "No fault codes" in result.report["health"]["summary"]


def test_pipeline_reports_skipped_checks(tmp_path, detector):
    path = tmp_path / "basic.csv"
    # steady cruise so the engine check (RPM jitter at steady speed) has data
    make_trip(9, 1, n=300).assign(speed_kmh=50.0)[["time_ms", "speed_kmh", "engine_rpm"]].to_csv(path, index=False)
    r = run(path, detector=detector)[0].report
    assert r["checks_run"]["Fuel system (fuel trims)"].startswith("not checked")
    assert r["checks_run"]["Engine (RPM stability)"] == "checked"
    assert any("not seen in training" in n for n in r["notes"])


def test_pipeline_without_model(tmp_path):
    path = tmp_path / "x.csv"
    make_trip(1, 1, n=20)[LOG_COLUMNS].to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        run(path, model_path=tmp_path / "missing.joblib")


def test_offline_html_embeds_plotly(rich_log, detector):
    log, dtc, _ = rich_log
    online = run(log, dtc_file=dtc, detector=detector)[0].html
    offline = run(log, dtc_file=dtc, detector=detector, offline_html=True)[0].html
    assert "cdn.plot.ly" in online and len(offline) > len(online) + 1_000_000


@pytest.mark.parametrize("seconds, expected", [(0, "0:00"), (65.4, "1:05"), (600, "10:00"), (None, "-")])
def test_fmt_time(seconds, expected):
    assert fmt_time(seconds) == expected
