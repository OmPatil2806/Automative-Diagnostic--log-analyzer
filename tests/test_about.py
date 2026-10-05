"""Tests for the About page and its templates."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from analyzer.ingestion.cleaner import clean_trip_log
from analyzer.ingestion.loader import REQUIRED_LOG_COLUMNS, load_trip_log
from analyzer.pipeline import collect_dtcs

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dashboard" / "assets"


def test_log_template_is_a_valid_log():
    logs = clean_trip_log(load_trip_log(ASSETS / "log_template.csv"))
    assert set(REQUIRED_LOG_COLUMNS) <= set(logs.columns)
    assert len(logs) == 4 and logs["time_ms"].is_monotonic_increasing


def test_dtc_template_is_valid():
    logs = clean_trip_log(load_trip_log(ASSETS / "log_template.csv"))
    events = collect_dtcs(logs, dtc_file=ASSETS / "dtc_template.csv")
    assert events["code"].tolist() == ["P0171"] and events["time_known"].all()


def _open_about():
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    return at.switch_page(str(ROOT / "dashboard" / "views" / "about.py")).run()


def test_about_page_renders():
    at = _open_about()
    assert not at.exception
    assert at.title[0].value == "About"
    assert len(at.tabs) == 5
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Log template (CSV)", "Fault codes template (CSV)", "Example trip with a fault (CSV)"]


def test_code_search_filters_the_reference():
    at = _open_about()
    full = len(at.dataframe[0].value)
    at.text_input(key="about_code_search").input("misfire").run()
    filtered = at.dataframe[0].value
    assert not at.exception
    assert 0 < len(filtered) < full
    assert filtered["Meaning"].str.contains("Misfire", case=False).any()
