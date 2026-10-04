"""Tests for the dashboard components (rendered headlessly with Streamlit's AppTest)."""

from streamlit.testing.v1 import AppTest

from dashboard.components.kpi import status_badge, status_counts


def test_status_badge_has_label_not_just_color():
    badge = status_badge("needs attention")
    assert "Needs attention" in badge and "#fab219" in badge
    assert "Unknown" in status_badge("unknown")


def test_status_counts():
    tiles = status_counts(["good", "good", "critical"])
    assert tiles == [("Good", 2, None), ("Needs attention", 0, None), ("Critical", 1, None)]


def _render_sample(key: str, include_dtcs: bool = True):
    from dashboard.components.report_view import report_view
    from dashboard.services.diagnosis_service import diagnose_sample

    report_view(diagnose_sample(key, include_dtcs)[0].report)


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def test_report_view_faulty_trip():
    at = AppTest.from_function(_render_sample, args=("vacuum_leak",), default_timeout=60).run()
    assert not at.exception
    text = _markdown(at)
    assert "75<small> / 100</small>" in text
    assert "Needs attention" in text
    assert "P0171" in text and "System Too Lean" in text
    assert "Likely cause:" in text and "Vacuum leak" in text
    assert "Inspect intake hoses" in text
    for title in ["Fault codes", "Anomalies and likely causes", "Recommended checks", "Checks run"]:
        assert f'class="dla-section">{title}<' in text


def test_report_view_normal_trip():
    at = AppTest.from_function(_render_sample, args=("normal_trip",), default_timeout=60).run()
    assert not at.exception
    text = _markdown(at)
    assert "Good" in text and "No fault codes." in text and "No unusual behaviour detected." in text


def test_report_view_early_warning():
    at = AppTest.from_function(_render_sample, args=("vacuum_leak", False), default_timeout=60).run()
    assert not at.exception
    assert "early warning, no fault code yet" in _markdown(at)


def test_kpi_row():
    def app():
        from dashboard.components.kpi import kpi_row
        kpi_row([("Trips", 7, None), ("Average score", 70.4, "mean health score")])

    at = AppTest.from_function(app).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Trips", "Average score"]
    assert at.metric[0].value == "7"
