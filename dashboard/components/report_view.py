"""Render a diagnosis report (the dict built by analyzer.reporting.report.build_report)."""

import html

import streamlit as st

from analyzer.reporting.report import fmt_time
from dashboard.components.kpi import status_badge
from dashboard.components.styles import inject_styles
from dashboard.theme import SEVERITY

e = html.escape


def trip_caption(report: dict) -> str:
    trip = report["trip"]
    when = f" · {trip['start'][:16]}" if trip["start"] else ""
    return (f"Vehicle {report['vehicle_id']} · Trip {report['trip_id']}{when} · "
            f"{fmt_time(trip['duration_s'])} min · {trip['distance_km']} km · {trip['readings']} readings")


def health_card(report: dict) -> None:
    """Large score, status badge and the one-paragraph summary."""
    inject_styles()
    h = report["health"]
    with st.container(border=True):
        left, right = st.columns([1, 3], vertical_alignment="center")
        left.markdown(f'<div class="dla-score">{h["score"]}<small> / 100</small></div>', unsafe_allow_html=True)
        left.markdown(status_badge(h["status"]), unsafe_allow_html=True)
        right.markdown(f"**Summary**  \n{e(h['summary'])}")


def fault_code_table(report: dict) -> None:
    if not report["dtcs"]:
        st.markdown('<span class="dla-muted">No fault codes.</span>', unsafe_allow_html=True)
        return
    rows = "".join(
        f"<tr><td><span class='dla-code'>{e(d['code'])}</span></td>"
        f"<td>{e(d['description'])}<br><span class='dla-muted'>{e(d['advice'])}</span></td>"
        f"<td><span class='dla-dot' style='background:{SEVERITY.get(d['severity'], '#7a7974')}'></span> "
        f"{e(d['severity'])}</td>"
        f"<td>{'set at ' + fmt_time(d['time_s']) if d['time_s'] is not None else 'read after drive'}</td></tr>"
        for d in report["dtcs"]
    )
    st.markdown(
        "<table class='dla-table'><thead><tr><th>Code</th><th>Meaning</th><th>Severity</th><th>When</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


def finding_card(index: int, finding: dict) -> None:
    tag = ("early warning, no fault code yet" if finding["early_warning"]
           else "linked to " + ", ".join(finding["dtc_codes"]))
    with st.container(border=True):
        st.markdown(f"**Anomaly {index}: {e(finding['subsystem'])}**")
        st.markdown(
            f"<span class='dla-chip'>{fmt_time(finding['start_s'])}–{fmt_time(finding['end_s'])}</span>"
            f"<span class='dla-chip'>{e(tag)}</span>"
            f"<span class='dla-chip'>peak score {finding['peak_score']:.1f}</span>",
            unsafe_allow_html=True,
        )
        warning = finding["warning_before_dtc_s"]
        if warning is not None and warning > 0:
            st.markdown(f"Detected **{warning:.0f} s before** the fault code.")
        st.markdown(f"**Likely cause:** {e(finding['cause_label'])} "
                    f"<span class='dla-muted'>({e(finding['confidence'])} confidence)</span>",
                    unsafe_allow_html=True)
        st.markdown("\n".join(f"- {e(x)}" for x in finding["evidence"]))


def findings(report: dict) -> None:
    if not report["findings"]:
        st.markdown('<span class="dla-muted">No unusual behaviour detected.</span>', unsafe_allow_html=True)
    for i, finding in enumerate(report["findings"], start=1):
        finding_card(i, finding)


def recommended_checks(report: dict) -> None:
    if report["recommended_checks"]:
        st.markdown("\n".join(f"{i}. {e(c)}" for i, c in enumerate(report["recommended_checks"], start=1)))
    else:
        st.markdown('<span class="dla-muted">None needed.</span>', unsafe_allow_html=True)
    if report.get("other_possible_causes"):
        st.caption("Other possible causes for these codes: " + ", ".join(report["other_possible_causes"]))


def checks_run(report: dict) -> None:
    rows = "".join(
        f"<tr><td>{e(name)}</td><td>{'Checked' if state == 'checked' else e(state.capitalize())}</td></tr>"
        for name, state in report["checks_run"].items()
    )
    st.markdown(f"<table class='dla-table'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    for note in report["notes"]:
        st.caption(f"Note: {note}")


def section(title: str) -> None:
    """Section heading sized between body text and the page title."""
    st.markdown(f'<div class="dla-section">{e(title)}</div>', unsafe_allow_html=True)


def report_view(report: dict) -> None:
    """The complete report: health, fault codes, anomalies, checks."""
    inject_styles()
    st.caption(trip_caption(report))
    health_card(report)

    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Fault codes")
        fault_code_table(report)
        section("Anomalies and likely causes")
        findings(report)
    with right:
        section("Recommended checks")
        recommended_checks(report)
        section("Checks run")
        checks_run(report)
