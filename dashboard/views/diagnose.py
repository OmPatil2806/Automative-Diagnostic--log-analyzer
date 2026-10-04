"""Diagnose a trip: pick a sample or upload a log, add fault codes, get the full report."""

import streamlit as st

from dashboard import state
from dashboard.components.downloads import trip_downloads
from dashboard.components.inputs import trip_request
from dashboard.components.layout import page_header
from dashboard.components.report_view import report_view
from dashboard.services.diagnosis_service import run_request, trip_labels
from dashboard.services.model_service import ModelNotFoundError

page_header(
    "Diagnose a trip",
    "Pick a sample trip or upload your own driving log, add the fault codes read from the vehicle, "
    "and get a diagnosis: health score, likely cause and what to check.",
)

with st.container(border=True):
    request, invalid_codes = trip_request()
    if invalid_codes:
        st.warning("Ignored entries that are not valid fault codes: " + ", ".join(invalid_codes))
    run = st.button("Diagnose", type="primary", key="diag_run",
                    disabled=request.source == "upload" and not request.log_bytes)

if run:
    try:
        results = run_request(request)
    except ModelNotFoundError as error:
        st.error(str(error))
        st.stop()
    except ValueError as error:
        st.error(f"Could not diagnose this log: {error}")
        st.stop()
    labels = trip_labels(request.label, results)
    for label, result in zip(labels, results):
        state.add_trip(label, result)
    state.set_current(labels)

current = state.get_current()
if not current:
    st.caption("Choose a trip above and press Diagnose.")
    st.stop()

label = current[0]
if len(current) > 1:
    label = st.selectbox(f"This file contains {len(current)} trips", current, key="diag_trip_choice")

result = state.get_trips()[label]
st.divider()
st.subheader(label)
with st.container(border=True):
    trip_downloads(result, label, key="diag_dl")
report_view(result.report)
