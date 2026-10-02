"""Project paths and settings (data folders, sensor columns, thresholds)."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VED_DIR = RAW_DIR / "ved"
VED_DYNAMIC_DIR = VED_DIR / "dynamic"
VED_STATIC_DIR = VED_DIR / "static"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Source files in the VED GitHub repository (https://github.com/gsoh/VED, Apache-2.0).
VED_BASE_URL = "https://raw.githubusercontent.com/gsoh/VED/master/Data/"
VED_DYNAMIC_ARCHIVES = ["VED_DynamicData_Part1.7z", "VED_DynamicData_Part2.7z"]
VED_STATIC_FILES = ["VED_Static_Data_ICE&HEV.xlsx", "VED_Static_Data_PHEV&EV.xlsx"]

# DayNum 1.0 in VED is 2017-11-01 00:00:00.
VED_EPOCH = pd.Timestamp("2017-11-01")

# Raw VED column name -> standard name used everywhere in this project.
VED_COLUMNS = {
    "DayNum": "day_num",
    "VehId": "vehicle_id",
    "Trip": "trip_id",
    "Timestamp(ms)": "time_ms",
    "Latitude[deg]": "latitude",
    "Longitude[deg]": "longitude",
    "Vehicle Speed[km/h]": "speed_kmh",
    "MAF[g/sec]": "maf_gs",
    "Engine RPM[RPM]": "engine_rpm",
    "Absolute Load[%]": "absolute_load_pct",
    "OAT[DegC]": "outside_air_temp_c",
    "Fuel Rate[L/hr]": "fuel_rate_lph",
    "Air Conditioning Power[kW]": "ac_power_kw",
    "Air Conditioning Power[Watts]": "ac_power_w",
    "Heater Power[Watts]": "heater_power_w",
    "HV Battery Current[A]": "hv_battery_current_a",
    "HV Battery SOC[%]": "hv_battery_soc_pct",
    "HV Battery Voltage[V]": "hv_battery_voltage_v",
    "Short Term Fuel Trim Bank 1[%]": "stft_b1_pct",
    "Short Term Fuel Trim Bank 2[%]": "stft_b2_pct",
    "Long Term Fuel Trim Bank 1[%]": "ltft_b1_pct",
    "Long Term Fuel Trim Bank 2[%]": "ltft_b2_pct",
}

KEY_COLUMNS = ["vehicle_id", "trip_id", "time_ms"]
SENSOR_COLUMNS = [c for c in VED_COLUMNS.values() if c not in KEY_COLUMNS + ["day_num"]]

# Physically plausible (min, max) per signal. Values outside are treated as sensor errors.
VALID_RANGES = {
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "speed_kmh": (0, 250),
    "maf_gs": (0, 500),
    "engine_rpm": (0, 8000),
    "absolute_load_pct": (0, 400),  # can exceed 100% on boosted engines
    "outside_air_temp_c": (-40, 60),
    "fuel_rate_lph": (0, 100),
    "ac_power_kw": (0, 10),
    "ac_power_w": (0, 10_000),
    "heater_power_w": (0, 10_000),
    "hv_battery_current_a": (-500, 500),
    "hv_battery_soc_pct": (0, 100),
    "hv_battery_voltage_v": (0, 1000),
    "stft_b1_pct": (-100, 100),
    "stft_b2_pct": (-100, 100),
    "ltft_b1_pct": (-100, 100),
    "ltft_b2_pct": (-100, 100),
}
