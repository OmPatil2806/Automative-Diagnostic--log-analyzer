"""Smoke tests for the Streamlit dashboard: every page renders without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "dashboard" / "app.py")
PAGES = sorted(p.name for p in (ROOT / "dashboard" / "pages").glob("*.py"))


def test_app_starts_on_overview():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Overview"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.switch_page(str(ROOT / "dashboard" / "pages" / page)).run()
    assert not at.exception, page
    assert at.title, f"{page} has no title"
