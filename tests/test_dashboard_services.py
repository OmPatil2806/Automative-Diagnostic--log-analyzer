"""Tests for the dashboard services (model loading, diagnosis, exports)."""

import io
import json
import zipfile

import pandas as pd
import pytest
from pypdf import PdfReader

from dashboard.services import diagnosis_service as ds
from dashboard.services import export_service as es
from dashboard.services.model_service import ModelNotFoundError, load_detector, model_info


# --- model ----------------------------------------------------------------

def test_load_detector_and_info():
    detector = load_detector()
    assert {"engine", "speed", "fuel"} <= set(detector.models)
    info = model_info()
    assert info["file"] == "anomaly_detector.joblib"
    assert info["vehicles_with_baseline"] > 0
    assert "of the last" in info["alarm_rule"]


def test_missing_model(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_detector(str(tmp_path / "missing.joblib"))


# --- samples and diagnosis ------------------------------------------------

def test_list_samples():
    samples = ds.list_samples()
    keys = [s.key for s in samples]
    assert keys[0] == "normal_trip" and len(keys) == 7
    normal = samples[0]
    assert normal.dtc_path is None and normal.log_path.exists()
    assert all(s.dtc_path and s.dtc_path.exists() for s in samples[1:])


@pytest.mark.parametrize("text, valid, invalid", [
    ("P0171, p0420  U0100", ["P0171", "P0420", "U0100"], []),
    ("P0171;P0171", ["P0171"], []),
    ("P0171 hello", ["P0171"], ["HELLO"]),
    ("", [], []),
])
def test_parse_dtc_text(text, valid, invalid):
    assert ds.parse_dtc_text(text) == (valid, invalid)


def test_diagnose_sample():
    result = ds.diagnose_sample("vacuum_leak")[0]
    assert result.report["main_finding"]["cause"] == "vacuum_leak"
    assert result.report["dtcs"][0]["code"] == "P0171"
    without_codes = ds.diagnose_sample("vacuum_leak", include_dtcs=False)[0]
    assert without_codes.report["dtcs"] == []
    assert without_codes.report["main_finding"]["early_warning"]


def test_diagnose_unknown_sample():
    with pytest.raises(ValueError, match="Unknown sample"):
        ds.diagnose_sample("nope")


def test_diagnose_upload_matches_sample():
    sample = next(s for s in ds.list_samples() if s.key == "misfire")
    results = ds.diagnose_upload(sample.log_path.read_bytes(), "my_misfire.csv",
                                 dtc_file_bytes=sample.dtc_path.read_bytes())
    report = results[0].report
    assert report["source"] == "my_misfire.csv"
    assert report["main_finding"]["cause"] == "misfire"
    assert report["dtcs"][0]["time_s"] is not None


def test_diagnose_upload_with_typed_codes():
    sample = next(s for s in ds.list_samples() if s.key == "rich_injector")
    report = ds.diagnose_upload(sample.log_path.read_bytes(), "trip.csv", dtc_codes=("P0172",))[0].report
    assert report["dtcs"][0]["code"] == "P0172" and report["dtcs"][0]["time_s"] is None


def test_diagnose_upload_bad_file():
    with pytest.raises(ValueError, match="missing required columns"):
        ds.diagnose_upload(b"a,b\n1,2\n", "bad.csv")


# --- exports --------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return ds.diagnose_sample("hv_battery_degradation")[0]


def test_safe_and_base_names(result):
    assert es.safe_name("Vacuum leak / trip 3") == "vacuum_leak_trip_3"
    assert es.safe_name("***") == "report"
    assert es.base_name(result) == f"hv_battery_degradation_vehicle{result.report['vehicle_id']}_trip{result.report['trip_id']}"


@pytest.mark.parametrize("fmt", list(es.FORMATS))
def test_trip_file_every_format(result, fmt):
    data = es.trip_file(result, fmt)
    assert data
    if fmt == "pdf":
        assert data.startswith(b"%PDF") and len(PdfReader(io.BytesIO(data)).pages) >= 1
    elif fmt == "html":
        assert data.startswith(b"<!doctype html>")
    elif fmt == "json":
        assert json.loads(data)["health"]["score"] == result.report["health"]["score"]
    elif fmt == "txt":
        assert b"HEALTH SCORE" in data
    elif fmt == "csv":
        readings = pd.read_csv(io.BytesIO(data))
        assert len(readings) == len(result.trip_logs)
        assert {"speed_kmh", "anomaly_score", "alarm", "score_battery"} <= set(readings.columns)


def test_trip_file_unknown_format(result):
    with pytest.raises(ValueError):
        es.trip_file(result, "docx")


def test_trip_bundle(result):
    names = zipfile.ZipFile(io.BytesIO(es.trip_bundle(result))).namelist()
    stem = es.base_name(result)
    assert sorted(names) == sorted([f"{stem}_report.pdf", f"{stem}_report.html", f"{stem}_report.json",
                                    f"{stem}_report.txt", f"{stem}_readings.csv"])


def test_fleet_summary_and_bundle():
    results = {s.label: ds.diagnose_sample(s.key)[0] for s in ds.list_samples()[:3]}
    summary = es.fleet_summary(results)
    assert len(summary) == 3
    assert summary["health_score"].is_monotonic_increasing
    assert summary.set_index("trip").loc["Normal trip", "status"] == "good"

    zf = zipfile.ZipFile(io.BytesIO(es.fleet_bundle(results, formats=("pdf", "json"))))
    names = zf.namelist()
    assert "summary.csv" in names
    assert sum(n.endswith(".pdf") for n in names) == 3 and sum(n.endswith(".json") for n in names) == 3
    assert len(pd.read_csv(zf.open("summary.csv"))) == 3


def test_timestamped():
    assert es.timestamped("fleet", "zip").startswith("fleet_") and es.timestamped("fleet", "zip").endswith(".zip")
