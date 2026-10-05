"""Saved model evaluation results (dashboard/assets/model_metrics.json, from scripts/export_metrics.py)."""

import json
from pathlib import Path

import streamlit as st

METRICS_PATH = Path(__file__).resolve().parents[1] / "assets" / "model_metrics.json"

FAULT_NAMES = {
    "vacuum_leak": "Vacuum leak",
    "maf_drift": "Dirty MAF sensor",
    "rich_injector": "Leaking injector",
    "misfire": "Engine misfire",
    "speed_sensor_failure": "Speed sensor failure",
    "hv_battery_degradation": "Hybrid battery wear",
}
GROUP_NAMES = {"engine": "Engine", "speed": "Speed sensor", "fuel": "Fuel system", "air": "Air intake",
               "battery": "Battery"}


@st.cache_data(show_spinner=False)
def load_metrics(path: str = str(METRICS_PATH)) -> dict | None:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
