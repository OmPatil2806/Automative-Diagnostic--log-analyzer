"""About: how the analyzer works, what input it expects, and templates to start from."""

from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer.analysis.health_score import DTC_PENALTY, MAX_ANOMALY_PENALTY
from analyzer.dtc.decoder import load_dtc_table
from dashboard.components.layout import page_header
from dashboard.components.report_view import html_table, section
from dashboard.services.diagnosis_service import SAMPLES_DIR

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPO_URL = "https://github.com/OmPatil2806/Automative-Diagnostic--log-analyzer"

page_header(
    "About",
    "How the analyzer works, what input it expects, and templates to start from.",
)

st.markdown(
    "The **Automotive Diagnostic Log Analyzer** reads a vehicle's OBD-II driving log, finds unusual behaviour with "
    "machine learning, explains the likely cause in plain language, links it to any fault codes the car stored, "
    "and scores the vehicle's health from 0 to 100. Because it watches the signals themselves, it can often warn "
    "**before** the engine computer sets a fault code."
)

tab_how, tab_input, tab_codes, tab_score, tab_data = st.tabs(
    ["How it works", "Input format", "Fault codes", "Health score", "Data, model and code"])

with tab_how:
    section("From a driving log to a diagnosis")
    steps = [
        ("1. Load and clean", "Read the CSV, standardise column names, sort by time, drop duplicates and "
                              "physically impossible values."),
        ("2. Score every reading", "Five detectors (engine, speed sensor, fuel, air intake, battery), each an "
                                   "Isolation Forest plus a range score, compare the signals with normal driving and "
                                   "with this vehicle's own baseline."),
        ("3. Find anomaly episodes", "An alarm needs the anomaly to persist (70% of the last 30 readings). "
                                     "Alarms are grouped into episodes with a suspect subsystem."),
        ("4. Explain the cause", "Rules read the evidence like a technician: fuel trims high and airflow normal "
                                 "means a vacuum leak; airflow low means a dirty MAF sensor."),
        ("5. Link fault codes", "Each fault code is matched to the anomaly in the same subsystem that came "
                                "before it; the gap is the early warning time."),
        ("6. Score and report", "Health score 0-100 and status, recommended checks, and a report as PDF, HTML, "
                                "JSON, text and CSV."),
    ]
    for row in (steps[:3], steps[3:]):
        for column, (title, text) in zip(st.columns(3), row):
            with column.container(border=True, height=225):
                st.markdown(f"**{title}**")
                st.caption(text)
    section("Where to go")
    html_table([
        {"page": "Overview", "use": "Health of every trip diagnosed in this session at a glance."},
        {"page": "Diagnose a trip", "use": "One trip: upload a log or pick a sample, add fault codes, get the report."},
        {"page": "Fleet reports", "use": "Many trips at once; download every report in one ZIP."},
        {"page": "Signal explorer", "use": "Plot any signal and see which detector raised the alarm, and when."},
        {"page": "Data exploration", "use": "What the VED driving data looks like and how it shaped the design."},
        {"page": "Model performance", "use": "How well, and how early, the model finds faults."},
    ], {"page": "Page", "use": "Use it for"})

with tab_input:
    section("Driving log (CSV)")
    st.markdown("One row per reading, about one per second. Only three columns are required; every optional signal "
                "turns on one more check. Raw VED column names (e.g. `Vehicle Speed[km/h]`) are also accepted.")
    html_table([
        {"col": "time_ms", "req": "Required", "unit": "milliseconds since the trip started", "use": "Timing"},
        {"col": "speed_kmh", "req": "Required", "unit": "km/h", "use": "Speed sensor check"},
        {"col": "engine_rpm", "req": "Required", "unit": "rpm", "use": "Engine (misfire) check"},
        {"col": "stft_b1_pct, ltft_b1_pct", "req": "Optional", "unit": "% fuel trim, bank 1",
         "use": "Fuel system (vacuum leak, rich running)"},
        {"col": "maf_gs, absolute_load_pct", "req": "Optional", "unit": "g/s, %", "use": "Air intake (MAF sensor)"},
        {"col": "hv_battery_voltage_v, hv_battery_current_a, hv_battery_soc_pct", "req": "Optional",
         "unit": "V, A, %", "use": "Hybrid / EV battery"},
        {"col": "vehicle_id", "req": "Optional", "unit": "VED vehicle number",
         "use": "Use that vehicle's learned baseline"},
        {"col": "trip_id", "req": "Optional", "unit": "number", "use": "Several trips in one file"},
    ], {"col": "Column", "req": "Needed", "unit": "Unit", "use": "Used for"})
    st.caption("Missing signals are fine: those checks are skipped and the report says so. A real car's data can be "
               "recorded with an ELM327 OBD-II adapter and the python-OBD library.")
    left, middle, right = st.columns(3)
    left.download_button("Log template (CSV)", (ASSETS / "log_template.csv").read_bytes(),
                         file_name="log_template.csv", mime="text/csv", icon=":material/download:",
                         on_click="ignore", key="about_log_template", width="stretch")
    middle.download_button("Fault codes template (CSV)", (ASSETS / "dtc_template.csv").read_bytes(),
                           file_name="dtc_template.csv", mime="text/csv", icon=":material/download:",
                           on_click="ignore", key="about_dtc_template", width="stretch")
    right.download_button("Example trip with a fault (CSV)", (SAMPLES_DIR / "vacuum_leak.csv").read_bytes(),
                          file_name="vacuum_leak.csv", mime="text/csv", icon=":material/download:",
                          on_click="ignore", key="about_example", width="stretch")

    section("Fault codes (optional)")
    html_table([
        {"how": "Type them", "example": "P0171, P0420", "note": "Codes read with a scan tool; the time is unknown, "
                                                               "so no early-warning time is shown."},
        {"how": "Upload a codes file", "example": "columns code, time_ms", "note": "Times enable the early-warning "
                                                                                 "measurement."},
        {"how": "Command line only: raw OBD-II response", "example": "\"43 01 71 04 20\"",
         "note": "Mode 03 hex from a scanner is decoded into codes."},
    ], {"how": "How", "example": "Example", "note": "Note"})
    st.caption("On the Fleet reports page, a codes file is matched to its log by name: `trip.csv` uses `trip_dtc.csv`.")

with tab_codes:
    section("Faults the analyzer can explain")
    html_table([
        {"fault": "Vacuum leak", "signals": "Fuel trims high, airflow normal", "code": "P0171"},
        {"fault": "Dirty or faulty MAF sensor", "signals": "Fuel trims high, airflow reading low", "code": "P0101"},
        {"fault": "Running rich (leaking injector)", "signals": "Fuel trims low", "code": "P0172"},
        {"fault": "Engine misfire", "signals": "RPM unstable at steady speed", "code": "P0300"},
        {"fault": "Faulty speed sensor", "signals": "Speed reads 0 while the car is moving", "code": "P0500"},
        {"fault": "Hybrid battery degradation", "signals": "Voltage below expected for charge and current", "code": "P0A7F"},
    ], {"fault": "Likely cause", "signals": "What the signals show", "code": "Typical code"})
    table = load_dtc_table()
    section(f"Fault code reference ({len(table)} codes)")
    codes = pd.DataFrame([{"Code": c, "Meaning": v["description"], "Severity": v["severity"],
                           "Possible causes": ", ".join(v["possible_causes"])} for c, v in table.items()])
    query = st.text_input("Search codes", placeholder="e.g. P0171, misfire, battery", key="about_code_search")
    if query:
        mask = codes.apply(lambda r: r.astype(str).str.contains(query, case=False, regex=False)).any(axis=1)
        codes = codes[mask]
    st.dataframe(codes, hide_index=True, width="stretch", height=360,
                 column_config={"Possible causes": st.column_config.TextColumn(width="large")})
    st.caption("Descriptions follow the SAE J2012 generic definitions; severity and causes were written for this "
               "project. Codes not in the table are still decoded by their structure.")

with tab_score:
    section("Health score")
    st.markdown("Every trip starts at **100**. Each fault code costs points by severity, and each anomaly that has "
                "**no** fault code yet (an early warning) costs points by how strong and how long it is.")
    html_table([{"what": f"Fault code, {sev} severity", "points": f"-{pts}"}
                for sev, pts in DTC_PENALTY.items() if sev != "unknown"]
               + [{"what": "Early warning (anomaly with no fault code)", "points": f"-10 to -25 each, at most "
                                                                           f"-{MAX_ANOMALY_PENALTY} in total"}],
               {"what": "Finding", "points": "Points"})
    section("Status")
    html_table([
        {"status": "Good", "range": "80-100", "meaning": "No action needed."},
        {"status": "Needs attention", "range": "50-79", "meaning": "Have it checked soon."},
        {"status": "Critical", "range": "below 50, or any critical fault code", "meaning": "Get it inspected before driving further."},
    ], {"status": "Status", "range": "Score", "meaning": "Meaning"})

with tab_data:
    section("Data")
    st.markdown(
        "Built on the **Vehicle Energy Dataset (VED)**: OBD-II logs from 383 personal cars (petrol, hybrid, plug-in "
        "hybrid and electric) driven in Ann Arbor, Michigan, from November 2017 to November 2018 "
        "([github.com/gsoh/VED](https://github.com/gsoh/VED), Apache License 2.0). VED has no fault codes, so "
        "faults are simulated on real trips to test the analyzer.\n\n"
        "G. Oh, D. J. LeBlanc, H. Peng, \"Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy "
        "Consumption Research,\" *IEEE Transactions on Intelligent Transportation Systems*, 2020."
    )
    section("Command line")
    st.code("python scripts/run_pipeline.py --log my_trip.csv --dtc P0171\n"
            "python scripts/run_pipeline.py --log my_trip.csv --dtc-file codes.csv --format pdf html",
            language="bash")
    section("Source code")
    st.markdown(f"[{REPO_URL.removeprefix('https://')}]({REPO_URL}) · Python, pandas, scikit-learn, Plotly, "
                "Streamlit, ReportLab · MIT License")
