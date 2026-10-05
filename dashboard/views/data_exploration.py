"""Data exploration: what the VED driving data looks like and how it shaped the analyzer."""

import pandas as pd
import streamlit as st

from dashboard.components import eda_charts as charts
from dashboard.components.kpi import kpi_row
from dashboard.components.layout import page_header
from dashboard.components.report_view import html_table, section
from dashboard.services.eda_service import load_eda_summary

page_header(
    "Data exploration",
    "The Vehicle Energy Dataset (VED): real OBD-II driving logs from 383 cars in Ann Arbor, Michigan. "
    "What the data looks like, and how each finding shaped the analyzer.",
)

eda = load_eda_summary()
if eda is None:
    st.warning("The data summary has not been generated. Download VED and run "
               "`python scripts/export_eda_summary.py`.")
    st.stop()

o = eda["overview"]
st.caption(f"First {eda['weeks']} weeks of VED ({o['first_day']} to {o['last_day']}), cleaned with the project's "
           f"own pipeline. Summary generated {eda['generated']}.")
kpi_row([
    ("Vehicles", o["vehicles"], None),
    ("Trips", f"{o['trips']:,}", None),
    ("Readings", f"{o['clean_readings'] / 1e6:.1f} M", f"{o['clean_readings']:,} after cleaning"),
    ("Median trip", f"{o['median_trip_min']} min", None),
    ("Reading interval", f"{o['median_gap_ms'] / 1000:.1f} s", f"Median gap; 90% of gaps are under "
                                                                f"{o['p90_gap_ms'] / 1000:.1f} s"),
])

tab_fleet, tab_signals, tab_values, tab_physics, tab_quality = st.tabs(
    ["Fleet and trips", "Which signals exist", "Signal values", "Signal relationships", "Data quality and findings"])

with tab_fleet:
    types = pd.DataFrame(eda["vehicle_types"])
    left, right = st.columns(2, gap="large")
    with left:
        section("Vehicles by type")
        st.plotly_chart(charts.hbar(types["name"].tolist(), types["vehicles"].tolist()), width="stretch",
                        key="eda_types")
    with right:
        section("Share of readings by type")
        st.plotly_chart(charts.hbar(types["name"].tolist(), types["readings_pct"].tolist(), fmt="{:.1f}%"),
                        width="stretch", key="eda_readings")
    section("Trip length")
    st.plotly_chart(charts.histogram(eda["trip_minutes"], "minutes (capped at 60)", show_band=False, height=260),
                    width="stretch", key="eda_trips")
    st.markdown("Most of the fleet is petrol cars and about a quarter are hybrids; there are only a few pure EVs. "
                "Trips are short (median **{} min**) and readings arrive about **once per second**: fine for "
                "fuel trims and battery voltage, too slow to see individual misfires.".format(o["median_trip_min"]))

with tab_signals:
    coverage = sorted(eda["coverage"], key=lambda c: -c["trips_pct"])
    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Share of trips that record each signal")
        st.plotly_chart(charts.hbar([c["name"] for c in coverage], [c["trips_pct"] for c in coverage], fmt="{:.0f}%"),
                        width="stretch", key="eda_coverage")
    with right:
        section("Logging changed after week 1")
        st.plotly_chart(charts.weekly_coverage(eda["weekly_coverage"]), width="stretch", key="eda_weekly")
        st.caption("Fuel trims were recorded in about 15% of trips in week 1, then 70-90%. The included model was "
                   "trained on week 1 only, so its fuel detector saw relatively little fuel-trim data.")
    section("Share of trips recording each signal, by vehicle type (%)")
    st.dataframe(pd.DataFrame(eda["coverage_by_type"]), width="stretch")
    st.markdown("Speed and RPM are almost always there; fuel trims exist only on combustion engines and battery "
                "signals only on plug-ins and EVs. No trip has everything, which is why the analyzer has **one "
                "detector per subsystem** and the report lists the checks it could not run.")

with tab_values:
    dists = eda["distributions"]
    choice = st.segmented_control("Signal", list(dists), default=list(dists)[0],
                                  format_func=lambda c: dists[c]["title"], key="eda_dist") or list(dists)[0]
    d = dists[choice]
    st.plotly_chart(charts.histogram(d, d["title"]), width="stretch", key="eda_hist")
    a, b, c = st.columns(3)
    a.metric("Median", f"{d['median']:g}")
    b.metric("1st percentile", f"{d['p1']:g}")
    c.metric("99th percentile", f"{d['p99']:g}")
    st.caption("Shaded: the central 98% of readings (1st to 99th percentile). Dotted line: median.")

    section("Every vehicle has its own normal")
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(charts.histogram(eda["per_vehicle"]["ltft_hist"], "usual long-term fuel trim per vehicle (%)",
                                         show_band=False, height=280),
                        width="stretch", key="eda_ltft")
    with right:
        st.plotly_chart(charts.vehicle_dots(eda["per_vehicle"]["hv_voltage_median"], "usual battery voltage per vehicle (V)"),
                        width="stretch", key="eda_hv")
    st.markdown("Each engine settles at its own fuel trim level and battery packs run at very different voltages. "
                "A fixed threshold would flag healthy cars or miss real faults, so the analyzer learns a "
                "**per-vehicle baseline** and measures deviation from it.")

with tab_physics:
    left, right = st.columns(2, gap="large")
    with left:
        section("Airflow follows RPM × load")
        st.plotly_chart(charts.airflow_scatter(eda["airflow"]), width="stretch", key="eda_airflow")
        st.caption("Each engine follows its own straight line. The analyzer learns one constant per vehicle and "
                   "flags a MAF sensor that reads well below it.")
    with right:
        battery = eda["battery"]
        section(f"Battery voltage vs current (vehicle {battery['vehicle_id']})")
        st.plotly_chart(charts.battery_scatter(battery), width="stretch", key="eda_battery")
        k = battery["coef"]
        st.caption(f"Fitted: V = {k[0]:.1f} + {k[1]:.2f} × SOC + {k[2]:.3f} × current, typical error "
                   f"±{battery['typical_error_v']:.1f} V. A battery sagging well below this line is degrading.")

with tab_quality:
    section("Data quality")
    html_table(eda["quality"], {"issue": "Issue", "count": "Count", "action": "Action"})
    st.caption(f"{o['raw_readings']:,} raw readings → {o['clean_readings']:,} after cleaning. VED is clean; the main "
               "challenge is missing signals, not wrong ones.")
    section("Findings and the design decisions they led to")
    html_table(eda["findings"], {"finding": "Finding", "decision": "Design decision"})
