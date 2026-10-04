"""Signal explorer: plot any signal of a diagnosed trip and see which detector raised the alarm."""

import streamlit as st

from dashboard import state
from dashboard.components.charts import (
    SIGNALS,
    available_signals,
    default_signals,
    detector_scores_figure,
    signals_figure,
)
from dashboard.components.inputs import load_samples_prompt
from dashboard.components.kpi import status_badge
from dashboard.components.layout import page_header
from dashboard.components.report_view import section, trip_caption
from dashboard.components.styles import inject_styles
from dashboard.services import export_service as es

page_header(
    "Signal explorer",
    "Plot any signal of a diagnosed trip, zoom into a time range, and see which detector raised the alarm, and when.",
)
inject_styles()

trips = state.get_trips()
if not trips:
    load_samples_prompt(key="explorer_load")
    st.stop()

labels = list(trips)
current = state.get_current()
label = st.selectbox("Trip", labels, index=labels.index(current[0]) if current else 0, key="explorer_trip")
result = trips[label]
logs, scored, report = result.trip_logs, result.trip_scored, result.report

main = report["main_finding"]
finding = f" · Likely cause: **{main['cause_label']}**" if main else " · No anomalies"
st.markdown(f"{status_badge(report['health']['status'])} &nbsp; Health score **{report['health']['score']}**{finding}",
            unsafe_allow_html=True)
st.caption(trip_caption(report))

duration = round((logs["time_ms"].max() - logs["time_ms"].min()) / 60_000, 1)
left, right = st.columns([3, 2], gap="large")
with left:
    chosen = st.multiselect("Signals", available_signals(logs), default=default_signals(logs, report),
                            format_func=SIGNALS.get, key=f"explorer_signals_{label}")
with right:
    window = st.slider("Time range (minutes into the trip)", 0.0, max(duration, 0.1), (0.0, max(duration, 0.1)),
                       step=0.1, key=f"explorer_window_{label}")

if chosen:
    st.plotly_chart(signals_figure(logs, report, chosen, window), width="stretch", key="explorer_signals_chart")
else:
    st.caption("Choose at least one signal to plot.")

section("Detector scores")
st.caption("Each subsystem detector scores every reading. Above 1.0 means outside normal behaviour; "
           "an alarm is raised when a detector stays above it. Shaded: anomaly periods. Dotted: fault code set.")
st.plotly_chart(detector_scores_figure(logs, scored, report, window), width="stretch", key="explorer_scores_chart")

with st.expander("Readings in this time range"):
    readings = es.scored_readings(result)
    minutes = (readings["time_ms"] - readings["time_ms"].min()) / 60_000
    selected = readings[minutes.between(*window)]
    st.dataframe(selected, hide_index=True, height=320)
    st.download_button("Download these readings (CSV)", selected.to_csv(index=False).encode("utf-8"),
                       file_name=f"{es.base_name(result, label)}_readings_{window[0]:.1f}-{window[1]:.1f}min.csv",
                       mime="text/csv", icon=":material/download:", on_click="ignore", key="explorer_dl")
