"""Run the diagnosis pipeline on sample trips or uploaded files (cached, no UI)."""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from analyzer.config import DATA_DIR
from analyzer.dtc.decoder import is_valid_code, normalize_code
from analyzer.pipeline import PipelineResult, run
from dashboard.services.model_service import load_detector

SAMPLES_DIR = DATA_DIR / "samples"

SAMPLE_INFO = {
    "normal_trip": ("Normal trip", "A healthy drive with no faults."),
    "vacuum_leak": ("Vacuum leak", "Extra air enters the engine; fuel trims rise. Sets P0171."),
    "maf_drift": ("Dirty MAF sensor", "The airflow sensor under-reads; fuel trims compensate. Sets P0101."),
    "rich_injector": ("Leaking injector", "The engine runs rich; fuel trims fall. Sets P0172."),
    "misfire": ("Engine misfire", "RPM becomes unstable at steady speed. Sets P0300."),
    "speed_sensor_failure": ("Speed sensor failure", "Speed drops to 0 while driving. Sets P0500."),
    "hv_battery_degradation": ("Hybrid battery wear", "Battery voltage sags under load. Sets P0A7F."),
}


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    description: str
    log_path: Path
    dtc_path: Path | None


def list_samples() -> list[Sample]:
    """Sample trips shipped with the repository, normal trip first."""
    samples = []
    for key, (label, description) in SAMPLE_INFO.items():
        log = SAMPLES_DIR / f"{key}.csv"
        if log.exists():
            dtc = SAMPLES_DIR / f"{key}_dtc.csv"
            samples.append(Sample(key, label, description, log, dtc if dtc.exists() else None))
    return samples


def parse_dtc_text(text: str) -> tuple[list[str], list[str]]:
    """Split free text like 'P0171, p0420 U0100' into (valid codes, invalid entries)."""
    entries = [normalize_code(t) for t in re.split(r"[\s,;]+", text or "") if t.strip()]
    valid = [c for c in dict.fromkeys(entries) if is_valid_code(c)]
    invalid = [c for c in entries if not is_valid_code(c)]
    return valid, invalid


@st.cache_data(show_spinner="Diagnosing the sample trip...", max_entries=32)
def diagnose_sample(key: str, include_dtcs: bool = True) -> list[PipelineResult]:
    """Diagnose one of the bundled sample trips, with its fault-code file if requested."""
    sample = next((s for s in list_samples() if s.key == key), None)
    if sample is None:
        raise ValueError(f"Unknown sample: {key}")
    dtc_file = sample.dtc_path if include_dtcs else None
    return run(sample.log_path, dtc_file=dtc_file, detector=load_detector())


@st.cache_data(show_spinner="Diagnosing the uploaded log...", max_entries=32)
def diagnose_upload(
    log_bytes: bytes,
    log_name: str,
    dtc_codes: tuple[str, ...] = (),
    dtc_file_bytes: bytes | None = None,
    vehicle_id: int | None = None,
) -> list[PipelineResult]:
    """Diagnose an uploaded CSV log (one result per trip in the file).

    The upload is written to a temporary folder under its own file name, so the
    report shows where it came from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / Path(log_name).name
        log_path.write_bytes(log_bytes)
        dtc_path = None
        if dtc_file_bytes:
            dtc_path = Path(tmp) / "fault_codes.csv"
            dtc_path.write_bytes(dtc_file_bytes)
        return run(log_path, dtc_codes=list(dtc_codes), dtc_file=dtc_path,
                   vehicle_id=vehicle_id, detector=load_detector())
