"""Tests for the Data exploration summary and page."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from analyzer.ingestion.cleaner import add_vehicle_info, clean_ved
from analyzer.ingestion.loader import load_ved, load_vehicle_info
from dashboard.services.eda_service import EDA_PATH, build_eda_summary, load_eda_summary, save_eda_summary

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"generated", "weeks", "overview", "vehicle_types", "trip_minutes", "coverage", "coverage_by_type",
        "weekly_coverage", "distributions", "per_vehicle", "airflow", "battery", "quality", "findings"}


@pytest.fixture
def small_summary(ved_dir, static_dir):
    raw = load_ved(ved_dir)
    clean = add_vehicle_info(clean_ved(raw), load_vehicle_info(static_dir))
    return build_eda_summary(raw, clean, weeks=["171101", "171108"])


def test_build_summary_on_small_data(small_summary):
    s = small_summary
    assert KEYS <= set(s)
    assert s["overview"]["raw_readings"] == 7 and s["overview"]["clean_readings"] == 5
    assert s["overview"]["vehicles"] == 2 and s["overview"]["trips"] == 3
    assert {q["issue"]: q["count"] for q in s["quality"]}["Rows with neither speed nor RPM"] == 1
    assert sum(t["vehicles"] for t in s["vehicle_types"]) == 2
    assert "speed_kmh" in s["distributions"]
    assert s["per_vehicle"]["ltft_hist"]["n"] == 0          # no fuel trims in this tiny log: empty, not an error
    json.dumps(s)                                   # JSON-ready


def test_save_and_load(small_summary, tmp_path):
    path = save_eda_summary(small_summary, tmp_path / "eda.json")
    assert load_eda_summary(str(path))["overview"] == small_summary["overview"]
    assert load_eda_summary(str(tmp_path / "missing.json")) is None


def test_committed_summary_is_complete_and_small():
    assert EDA_PATH.exists(), "run scripts/export_eda_summary.py"
    assert EDA_PATH.stat().st_size < 500_000
    s = json.loads(EDA_PATH.read_text())
    assert KEYS <= set(s)
    assert s["overview"]["vehicles"] > 100
    assert len(s["weekly_coverage"]) == s["weeks"]


def test_page_renders_every_tab():
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    at.switch_page(str(ROOT / "dashboard" / "views" / "data_exploration.py")).run()
    assert not at.exception
    assert at.title[0].value == "Data exploration"
    assert len(at.tabs) == 5
    assert len(at.get("plotly_chart")) == 10
    assert [m.label for m in at.metric][:5] == ["Vehicles", "Trips", "Readings", "Median trip", "Reading interval"]
    at.segmented_control(key="eda_dist").set_value("hv_battery_voltage_v").run()
    assert not at.exception
