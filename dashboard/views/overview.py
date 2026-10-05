"""Overview: health of every trip diagnosed in this session at a glance."""

import streamlit as st

from dashboard import state
from dashboard.components.charts import causes_figure, health_scores_figure, status_figure
from dashboard.components.inputs import load_samples_prompt
from dashboard.components.kpi import kpi_row
from dashboard.components.layout import page_header
from dashboard.components.report_view import section
from dashboard.services import export_service as es
from dashboard.theme import STATUS

page_header(
    "Overview",
    "Health of every trip diagnosed in this session: status, scores, fault codes and likely causes at a glance.",
)

trips = state.get_trips()
if not trips:
    load_samples_prompt(key="overview_load")
    section("What you can do here")
    cols = st.columns(3)
    for col, (page, title, text, icon) in zip(cols, [
        ("views/diagnose.py", "Diagnose a trip", "Upload a driving log or pick a sample and get a full report.",
         ":material/troubleshoot:"),
        ("views/fleet_reports.py", "Fleet reports", "Diagnose many trips at once and download every report.",
         ":material/folder_zip:"),
        ("views/signal_explorer.py", "Signal explorer", "See which signal changed and which detector caught it.",
         ":material/monitoring:"),
    ]):
        with col.container(border=True):
            st.page_link(page, label=title, icon=icon)
            st.caption(text)
    st.stop()

summary = es.fleet_summary(trips)
needs_work = summary[summary["status"] != "good"]
kpi_row([
    ("Trips", len(summary), "Trips diagnosed in this session"),
    ("Vehicles", summary["vehicle_id"].nunique(), None),
    ("Average score", f"{summary['health_score'].mean():.0f}", "Mean of the 0-100 health scores"),
    ("Need attention", len(needs_work), "Trips rated Needs attention or Critical"),
    ("Early warnings", int(summary["early_warnings"].sum()), "Anomalies found with no fault code set yet"),
])

left, right = st.columns([2, 3], gap="large")
with left:
    section("Trips by status")
    st.plotly_chart(status_figure(summary), width="stretch", key="overview_status")
    section("Likely causes found")
    if (summary["likely_cause"] != "").any():
        st.plotly_chart(causes_figure(summary), width="stretch", key="overview_causes")
    else:
        st.caption("No anomalies found in any trip.")
with right:
    section("Health score per trip")
    st.plotly_chart(health_scores_figure(summary), width="stretch", key="overview_scores")

section("All trips")
st.dataframe(
    summary.assign(status=summary["status"].map(lambda s: STATUS.get(s, {}).get("label", s))),
    hide_index=True,
    column_order=["trip", "health_score", "status", "fault_codes", "likely_cause", "early_warnings",
                  "duration_min", "distance_km"],
    column_config={
        "trip": st.column_config.TextColumn("Trip", width="medium"),
        "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d"),
        "status": st.column_config.TextColumn("Status"),
        "fault_codes": st.column_config.TextColumn("Fault codes"),
        "likely_cause": st.column_config.TextColumn("Likely cause", width="large"),
        "early_warnings": st.column_config.NumberColumn("Early warnings"),
        "duration_min": st.column_config.NumberColumn("Minutes", format="%.1f"),
        "distance_km": st.column_config.NumberColumn("km", format="%.1f"),
    },
)

left, middle, right = st.columns(3)
left.page_link("views/fleet_reports.py", label="Download all reports (Fleet reports)", icon=":material/folder_zip:")
middle.page_link("views/signal_explorer.py", label="Explore the signals", icon=":material/monitoring:")
if right.button("Clear this session", icon=":material/delete_sweep:", key="overview_clear"):
    state.clear_trips()
    st.rerun()
