"""Tests for the PDF report."""

import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from analyzer.detection.anomaly import AnomalyDetector
from analyzer.pipeline import run
from analyzer.reporting.pdf_report import render_pdf, signal_chart_png
from analyzer.synthetic.faults import FAULTS
from analyzer.synthetic.generator import inject_fault
from tests.conftest import make_trip

LOG_COLUMNS = ["time_ms", "speed_kmh", "engine_rpm", "maf_gs", "absolute_load_pct",
               "stft_b1_pct", "ltft_b1_pct", "hv_battery_voltage_v", "hv_battery_current_a", "hv_battery_soc_pct"]


@pytest.fixture(scope="module")
def detector() -> AnomalyDetector:
    normal = pd.concat([make_trip(1, t, n=300, seed=t) for t in range(30)], ignore_index=True)
    return AnomalyDetector(n_estimators=50).fit(normal)


@pytest.fixture(scope="module")
def faulty_result(detector, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pdf")
    trip, label = inject_fault(make_trip(1, 7, n=300, seed=7), FAULTS["rich_injector"], np.random.default_rng(0))
    log, dtc = tmp / "rich.csv", tmp / "rich_dtc.csv"
    trip[["vehicle_id"] + LOG_COLUMNS].to_csv(log, index=False)
    pd.DataFrame({"code": ["P0172"], "time_ms": [label["dtc_time_ms"]]}).to_csv(dtc, index=False)
    return run(log, dtc_file=dtc, detector=detector)[0]


def pdf_text(data: bytes) -> str:
    """All text in the PDF with whitespace collapsed (lines wrap wherever the layout puts them)."""
    reader = PdfReader(io.BytesIO(data))
    return " ".join(" ".join(page.extract_text() for page in reader.pages).split())


def test_pdf_is_valid_and_has_metadata(faulty_result):
    data = faulty_result.pdf()
    assert data.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(data))
    assert 1 <= len(reader.pages) <= 3
    assert "Diagnosis report" in reader.metadata.title


def test_pdf_contains_report_content(faulty_result):
    text = pdf_text(faulty_result.pdf())
    report = faulty_result.report
    assert "Vehicle diagnosis report" in text
    assert f"{report['health']['score']}" in text
    assert "NEEDS ATTENTION" in text
    assert "P0172" in text and "System Too Rich" in text
    assert report["main_finding"]["cause_label"].split(" (")[0] in text
    assert report["recommended_checks"][0] in text
    assert "Checks run" in text and "Page 1" in text


def test_pdf_for_normal_trip(tmp_path, detector):
    path = tmp_path / "normal.csv"
    make_trip(1, 3, n=300, seed=303)[["vehicle_id"] + LOG_COLUMNS].to_csv(path, index=False)
    text = pdf_text(run(path, detector=detector)[0].pdf())
    assert "GOOD" in text
    assert "No fault codes." in text
    assert "No unusual behaviour detected." in text


def test_chart_png(faulty_result):
    png = signal_chart_png(faulty_result.report, faulty_result.trip_logs, faulty_result.trip_scored)
    assert png.startswith(b"\x89PNG")


def test_render_pdf_escapes_markup(faulty_result):
    report = {**faulty_result.report, "source": "a<b>&c.csv"}
    report["health"] = {**report["health"], "summary": "Use <care> & check"}
    text = pdf_text(render_pdf(report, faulty_result.trip_logs, faulty_result.trip_scored))
    assert "Use <care> & check" in text
