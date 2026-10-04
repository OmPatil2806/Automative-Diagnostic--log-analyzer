"""Input widgets for choosing a trip and its fault codes. Returns a DiagnosisRequest (no diagnosis here)."""

import streamlit as st

from dashboard.services.diagnosis_service import DiagnosisRequest, list_samples, parse_dtc_text

DEFAULT_SAMPLE = "vacuum_leak"
REQUIRED = "`time_ms`, `speed_kmh`, `engine_rpm`"


def _sample_inputs() -> DiagnosisRequest:
    samples = list_samples()
    keys = [s.key for s in samples]
    by_key = {s.key: s for s in samples}
    key = st.selectbox(
        "Sample trip",
        keys,
        index=keys.index(DEFAULT_SAMPLE) if DEFAULT_SAMPLE in keys else 0,
        format_func=lambda k: by_key[k].label,
        key="diag_sample",
    )
    sample = by_key[key]
    st.caption(sample.description)
    include = False
    if sample.dtc_path:
        include = st.checkbox(
            "Include the fault code this trip set",
            value=True,
            key="diag_sample_codes",
            help="Untick to see how the analyzer warns from the signals alone, before any fault code.",
        )
    return DiagnosisRequest(source="sample", sample_key=key, include_sample_dtcs=include)


def _upload_inputs() -> tuple[DiagnosisRequest, list[str]]:
    log = st.file_uploader(
        "Driving log (CSV)",
        type=["csv"],
        key="diag_upload",
        help=f"One row per reading. Required columns: time_ms, speed_kmh, engine_rpm. "
             "Optional: maf_gs, absolute_load_pct, stft_b1_pct, ltft_b1_pct, hv_battery_voltage_v, "
             "hv_battery_current_a, hv_battery_soc_pct, vehicle_id.",
    )
    st.caption(f"Required columns: {REQUIRED}. Any other signals are optional.")

    mode = st.radio("Fault codes", ["None", "Type codes", "Upload a codes file"], horizontal=True, key="diag_dtc_mode")
    codes, invalid, dtc_bytes = (), [], None
    if mode == "Type codes":
        text = st.text_input("Codes read from the vehicle", placeholder="e.g. P0171, P0420", key="diag_dtc_text")
        valid, invalid = parse_dtc_text(text)
        codes = tuple(valid)
    elif mode == "Upload a codes file":
        file = st.file_uploader("Fault codes (CSV with columns code, time_ms)", type=["csv"], key="diag_dtc_file")
        dtc_bytes = file.getvalue() if file else None

    vehicle_id = st.number_input(
        "Vehicle ID (optional)", min_value=0, step=1, value=None, key="diag_vehicle",
        help="A VED vehicle ID lets the analyzer use that vehicle's learned baseline. Leave empty for any other car.",
    )
    request = DiagnosisRequest(
        source="upload",
        log_bytes=log.getvalue() if log else None,
        log_name=log.name if log else None,
        dtc_codes=codes,
        dtc_file_bytes=dtc_bytes,
        vehicle_id=int(vehicle_id) if vehicle_id is not None else None,
    )
    return request, invalid


def trip_request() -> tuple[DiagnosisRequest, list[str]]:
    """Source picker plus its inputs. Returns the request and any invalid typed codes."""
    source = st.segmented_control(
        "Data source", ["Sample trip", "Upload a CSV"], default="Sample trip", key="diag_source",
    ) or "Sample trip"
    if source == "Sample trip":
        return _sample_inputs(), []
    return _upload_inputs()
