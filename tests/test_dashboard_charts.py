"""Tests for the dashboard charts."""

import pytest

from dashboard.components import charts
from dashboard.services.diagnosis_service import diagnose_sample


@pytest.fixture(scope="module")
def result():
    return diagnose_sample("vacuum_leak")[0]


def test_available_and_default_signals(result):
    available = charts.available_signals(result.trip_logs)
    assert {"speed_kmh", "engine_rpm", "total_fuel_trim", "stft_b1_pct"} <= set(available)
    assert "hv_battery_voltage_v" not in available           # not in this log
    assert charts.default_signals(result.trip_logs, result.report) == ["speed_kmh", "engine_rpm", "total_fuel_trim"]


def test_signals_figure_panels_and_markers(result):
    fig = charts.signals_figure(result.trip_logs, result.report, ["speed_kmh", "total_fuel_trim"])
    assert [t.name for t in fig.data] == ["Speed (km/h)", "Total fuel trim (%)"]
    shapes = fig.layout.shapes
    assert any(s.type == "rect" for s in shapes)              # anomaly shading
    assert any(s.type == "line" for s in shapes)              # fault code line
    assert any("P0171 set" in (a.text or "") for a in fig.layout.annotations)


def test_signals_figure_time_window(result):
    fig = charts.signals_figure(result.trip_logs, result.report, ["speed_kmh"], window=(1.0, 2.0))
    xs = fig.data[0].x
    assert min(xs) >= 1.0 and max(xs) <= 2.0
    assert not any(s.type == "line" for s in fig.layout.shapes)   # P0171 (at ~8 min) is outside the window


def test_detector_scores_figure(result):
    fig = charts.detector_scores_figure(result.trip_logs, result.trip_scored, result.report)
    names = [t.name for t in fig.data]
    assert "Fuel system" in names and "Engine" in names
    colors = {t.name: t.line.color for t in fig.data}
    assert colors["Engine"] == charts.DETECTOR_COLORS["engine"]
    assert fig.layout.showlegend


def test_trip_overview_shading_starts_at_trip_time(result):
    shifted = result.trip_logs.assign(time_ms=result.trip_logs["time_ms"] + 600_000)
    scored = result.trip_scored.assign(time_ms=result.trip_scored["time_ms"] + 600_000)
    fig = charts.trip_overview(shifted, scored, result.report)
    # time axis is relative to the trip start, so it still begins at 0 when the log does not
    assert min(fig.data[0].x) == 0
