"""Tests for the saved model metrics and the Model performance page."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.services.metrics_service import FAULT_NAMES, METRICS_PATH, load_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_committed_metrics_are_complete():
    assert METRICS_PATH.exists(), "run scripts/export_metrics.py"
    m = json.loads(METRICS_PATH.read_text())
    assert {"model", "dataset", "methods", "per_fault", "warning_times", "persistence", "diagnosis"} <= set(m)
    assert set(m["methods"]) == {"forest", "range", "combined"}
    assert [f["fault"] for f in m["per_fault"]] == list(FAULT_NAMES)
    assert sum(g["used"] for g in m["persistence"]) == 1
    # the combined score is the one used, and it is at least as good as either part
    assert m["methods"]["combined"]["f1"] >= max(m["methods"]["forest"]["f1"], m["methods"]["range"]["f1"])


def test_load_metrics_missing(tmp_path):
    assert load_metrics(str(tmp_path / "none.json")) is None


def test_page_renders_every_tab():
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    at.switch_page(str(ROOT / "dashboard" / "views" / "model_performance.py")).run()
    assert not at.exception
    assert at.title[0].value == "Model performance"
    assert len(at.tabs) == 5
    assert [x.label for x in at.metric][:6] == ["Faulty trips detected", "Normal trips with a false alarm",
                                                 "Right root cause", "F1 score", "Precision", "Recall"]
    assert len(at.get("plotly_chart")) == 4
