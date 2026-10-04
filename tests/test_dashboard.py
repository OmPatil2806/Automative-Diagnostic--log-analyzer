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
