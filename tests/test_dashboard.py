"""Smoke tests for the Streamlit dashboard: every page renders without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "dashboard" / "app.py")
PAGES = sorted(p.name for p in (ROOT / "dashboard" / "views").glob("*.py"))


def test_no_pages_folder_next_to_app():
    # Streamlit would auto-load a pages/ folder as separate pages that skip app.py's setup,
    # which breaks opening a page URL directly (e.g. /diagnose). Pages live in views/.
    assert not (ROOT / "dashboard" / "pages").exists()


def test_app_starts_on_overview():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Overview"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.switch_page(str(ROOT / "dashboard" / "views" / page)).run()
    assert not at.exception, page
    assert at.title, f"{page} has no title"


def _open_diagnose():
    at = AppTest.from_file(APP, default_timeout=60).run()
    return at.switch_page(str(ROOT / "dashboard" / "views" / "diagnose.py")).run()


def test_diagnose_sample_end_to_end():
    at = _open_diagnose()
    assert at.selectbox(key="diag_sample").value == "vacuum_leak"
    at.button(key="diag_run").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "P0171" in text and "Vacuum leak" in text and "Needs attention" in text
    assert at.subheader[0].value == "Vacuum leak"
    assert "Vacuum leak" in at.session_state["diagnosed_trips"]


def test_diagnose_sample_without_codes_shows_early_warning():
    at = _open_diagnose()
    at.selectbox(key="diag_sample").set_value("hv_battery_degradation")
    at.checkbox(key="diag_sample_codes").uncheck()
    at.button(key="diag_run").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "early warning, no fault code yet" in text and "Hybrid battery degradation" in text


def test_diagnose_result_survives_page_switch():
    at = _open_diagnose()
    at.button(key="diag_run").click().run()
    at.switch_page(str(ROOT / "dashboard" / "views" / "about.py")).run()
    at.switch_page(str(ROOT / "dashboard" / "views" / "diagnose.py")).run()
    assert at.subheader[0].value == "Vacuum leak"


def test_upload_mode_waits_for_a_file():
    at = _open_diagnose()
    at.segmented_control(key="diag_source").set_value("Upload a CSV").run()
    assert not at.exception
    assert at.button(key="diag_run").disabled


def test_diagnose_shows_all_download_buttons():
    at = _open_diagnose()
    at.button(key="diag_run").click().run()
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download all formats (ZIP)", "PDF", "HTML", "JSON", "Text", "Readings CSV"]



def test_diagnose_shows_signal_chart():
    at = _open_diagnose()
    at.button(key="diag_run").click().run()
    assert len(at.get("plotly_chart")) == 1


def _open_explorer():
    at = AppTest.from_file(APP, default_timeout=120).run()
    return at.switch_page(str(ROOT / "dashboard" / "views" / "signal_explorer.py")).run()


def test_explorer_empty_state_loads_samples():
    at = _open_explorer()
    assert not at.exception
    assert at.button(key="explorer_load")
    at.button(key="explorer_load").click().run()
    assert not at.exception
    assert len(at.session_state["diagnosed_trips"]) == 7
    assert at.selectbox(key="explorer_trip").options[0] == "Normal trip"
    assert len(at.get("plotly_chart")) == 2   # signals + detector scores


def test_explorer_defaults_follow_the_trip():
    at = _open_explorer()
    at.button(key="explorer_load").click().run()
    at.selectbox(key="explorer_trip").set_value("Vacuum leak").run()
    signals = at.multiselect(key="explorer_signals_Vacuum leak").value
    assert signals == ["speed_kmh", "engine_rpm", "total_fuel_trim"]
    at.multiselect(key="explorer_signals_Vacuum leak").set_value([]).run()
    assert not at.exception and len(at.get("plotly_chart")) == 1   # only detector scores left


def _open_fleet():
    at = AppTest.from_file(APP, default_timeout=180).run()
    return at.switch_page(str(ROOT / "dashboard" / "views" / "fleet_reports.py")).run()


def test_fleet_diagnoses_all_samples():
    at = _open_fleet()
    assert len(at.multiselect(key="fleet_samples").value) == 7
    at.button(key="fleet_run").click().run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Trips", "Average score"]
    assert at.metric[0].value == "7"
    table = at.dataframe[0].value
    assert len(table) == 7 and table["health_score"].is_monotonic_increasing
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download all 7 reports (ZIP)", "Download summary (CSV)"]
    assert len(at.session_state["diagnosed_trips"]) == 7


def test_fleet_subset_and_session_source():
    at = _open_fleet()
    at.multiselect(key="fleet_samples").set_value(["normal_trip", "misfire"])
    at.button(key="fleet_run").click().run()
    assert at.metric[0].value == "2"
    at.segmented_control(key="fleet_source").set_value("Trips in this session").run()
    at.button(key="fleet_run").click().run()
    assert not at.exception and at.metric[0].value == "2"


def test_fleet_upload_waits_for_files():
    at = _open_fleet()
    at.segmented_control(key="fleet_source").set_value("Upload CSV files").run()
    assert at.button(key="fleet_run").disabled
