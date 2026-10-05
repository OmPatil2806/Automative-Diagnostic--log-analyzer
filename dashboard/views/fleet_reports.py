"""Fleet reports: diagnose many trips at once and download every report in one ZIP."""

import streamlit as st

from dashboard import state
from dashboard.components.kpi import kpi_row, status_counts
from dashboard.components.layout import page_header
from dashboard.components.report_view import section
from dashboard.services import export_service as es
from dashboard.services.diagnosis_service import diagnose_batch, diagnose_samples, list_samples
from dashboard.theme import STATUS

page_header(
    "Fleet reports",
    "Diagnose several trips at once, compare their health, and download every report in one ZIP file.",
)

SOURCES = ["Sample trips", "Upload CSV files", "Trips in this session"]

with st.container(border=True):
    source = st.segmented_control("Trips to diagnose", SOURCES, default=SOURCES[0], key="fleet_source") or SOURCES[0]
    ready, files, keys, include = True, [], [], True

    if source == "Sample trips":
        samples = list_samples()
        labels = {s.key: s.label for s in samples}
        keys = st.multiselect("Samples", list(labels), default=list(labels), format_func=labels.get, key="fleet_samples")
        include = st.checkbox("Include each trip's fault code", value=True, key="fleet_codes")
        ready = bool(keys)
    elif source == "Upload CSV files":
        uploads = st.file_uploader("Driving logs (CSV), several at once", type=["csv"],
                                   accept_multiple_files=True, key="fleet_upload")
        st.caption("Fault-code files are matched to logs by name: `trip.csv` uses `trip_dtc.csv` "
                   "(columns `code`, `time_ms`). Logs without one are diagnosed from their signals alone.")
        files = [(f.name, f.getvalue()) for f in uploads or []]
        ready = bool(files)
    else:
        in_session = state.get_trips()
        st.caption(f"{len(in_session)} trip(s) diagnosed in this session so far (Diagnose page, Signal explorer, "
                   "or earlier batches)." if in_session else "No trips diagnosed in this session yet.")
        ready = bool(in_session)

    run = st.button("Diagnose all" if source != SOURCES[2] else "Show these trips", type="primary",
                    key="fleet_run", disabled=not ready)

if run:
    errors = {}
    with st.spinner("Diagnosing trips..."):
        if source == "Sample trips":
            results = diagnose_samples(keys, include)
        elif source == "Upload CSV files":
            results, errors = diagnose_batch(files)
        else:
            results = dict(state.get_trips())
    for label, result in results.items():
        state.add_trip(label, result)
    state.set_fleet(list(results))
    for name, message in errors.items():
        st.error(f"**{name}**: {message}")

fleet = state.get_fleet()
if not fleet:
    st.caption("Choose trips above and press the button.")
    st.stop()

summary = es.fleet_summary(fleet)
st.divider()
section(f"{len(summary)} trips")
kpi_row([("Trips", len(summary), None),
         ("Average score", f"{summary['health_score'].mean():.0f}", "Mean of the trips' 0-100 health scores"),
         *status_counts(summary["status"].tolist())])

st.dataframe(
    summary.assign(status=summary["status"].map(lambda s: STATUS.get(s, {}).get("label", s))),
    hide_index=True,
    column_order=["trip", "health_score", "status", "fault_codes", "likely_cause", "warning_before_code_s",
                  "anomalies", "early_warnings", "duration_min", "distance_km", "vehicle_id", "trip_id"],
    column_config={
        "trip": st.column_config.TextColumn("Trip", width="medium"),
        "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d"),
        "status": st.column_config.TextColumn("Status"),
        "fault_codes": st.column_config.TextColumn("Fault codes"),
        "likely_cause": st.column_config.TextColumn("Likely cause", width="large"),
        "warning_before_code_s": st.column_config.NumberColumn("Warning before code (s)", format="%.0f"),
        "anomalies": st.column_config.NumberColumn("Anomalies"),
        "early_warnings": st.column_config.NumberColumn("Early warnings"),
        "duration_min": st.column_config.NumberColumn("Minutes", format="%.1f"),
        "distance_km": st.column_config.NumberColumn("km", format="%.1f"),
        "vehicle_id": st.column_config.NumberColumn("Vehicle", format="%d"),
        "trip_id": st.column_config.NumberColumn("Trip ID", format="%d"),
    },
)

section("Download")
with st.container(border=True):
    formats = st.multiselect("Report formats in the ZIP", list(es.FORMATS), default=["pdf", "html", "json"],
                             format_func=lambda f: es.FORMATS[f][0], key="fleet_formats")
    left, right = st.columns(2)
    with left:
        st.download_button(
            f"Download all {len(fleet)} reports (ZIP)",
            data=lambda: es.fleet_bundle(fleet, formats=tuple(formats)),
            file_name=es.timestamped("fleet_reports", "zip"), mime="application/zip",
            icon=":material/folder_zip:", type="primary", on_click="ignore", key="fleet_zip",
            disabled=not formats, width="stretch",
            help="summary.csv plus one folder per trip with the chosen report formats",
        )
    with right:
        st.download_button(
            "Download summary (CSV)", data=summary.to_csv(index=False).encode("utf-8"),
            file_name=es.timestamped("fleet_summary", "csv"), mime="text/csv",
            icon=":material/table_view:", on_click="ignore", key="fleet_csv", width="stretch",
        )
    st.caption("Building PDFs takes about a second per trip, so large batches can take a moment.")
