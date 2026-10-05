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


@dataclass(frozen=True)
class DiagnosisRequest:
    """Everything the user chose on the Diagnose page."""

    source: str                         # "sample" or "upload"
    sample_key: str | None = None
    include_sample_dtcs: bool = True
    log_bytes: bytes | None = None
    log_name: str | None = None
    dtc_codes: tuple[str, ...] = ()
    dtc_file_bytes: bytes | None = None
    vehicle_id: int | None = None

    @property
    def label(self) -> str:
        """Display name used for the trip in this session."""
        if self.source == "sample":
            return SAMPLE_INFO[self.sample_key][0] if self.sample_key in SAMPLE_INFO else str(self.sample_key)
        return self.log_name or "Uploaded log"


def run_request(request: DiagnosisRequest) -> list[PipelineResult]:
    """Run a request from the Diagnose page. Raises ValueError with a readable message on bad input."""
    if request.source == "sample":
        return diagnose_sample(request.sample_key, request.include_sample_dtcs)
    if request.source == "upload":
        if not request.log_bytes:
            raise ValueError("Upload a driving log CSV first.")
        return diagnose_upload(request.log_bytes, request.log_name or "uploaded.csv",
                               request.dtc_codes, request.dtc_file_bytes, request.vehicle_id)
    raise ValueError(f"Unknown source: {request.source}")


def trip_labels(label: str, results: list[PipelineResult]) -> list[str]:
    """One name per trip: the label itself, or 'label · trip N' when a file holds several trips."""
    if len(results) == 1:
        return [label]
    return [f"{label} · trip {r.report['trip_id']}" for r in results]


def diagnose_all_samples() -> dict[str, PipelineResult]:
    """Every bundled sample trip with its fault codes, keyed by sample label."""
    return {s.label: diagnose_sample(s.key)[0] for s in list_samples()}


DTC_SUFFIX = "_dtc.csv"


def pair_uploads(files: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes, bytes | None]], list[str]]:
    """Match logs with their fault-code files by name: trip.csv <-> trip_dtc.csv.

    Returns (log name, log bytes, codes bytes or None) per log, and the names of
    code files that have no matching log.
    """
    logs = {n: b for n, b in files if not n.lower().endswith(DTC_SUFFIX)}
    codes = {n[: -len(DTC_SUFFIX)].lower(): b for n, b in files if n.lower().endswith(DTC_SUFFIX)}
    pairs = [(name, data, codes.pop(Path(name).stem.lower(), None)) for name, data in logs.items()]
    unmatched = [n for n, _ in files if n.lower().endswith(DTC_SUFFIX) and n[: -len(DTC_SUFFIX)].lower() in codes]
    return pairs, unmatched


def short_error(file_name: str, message: str) -> str:
    """Trim pipeline errors for display: drop the repeated file name and the column reference list."""
    message = message.removeprefix(f"{file_name} ").removeprefix("is ").split(". Required:")[0]
    message = message.replace("[", "").replace("]", "").replace("'", "")
    return message[:1].upper() + message[1:]


def diagnose_batch(files: list[tuple[str, bytes]]) -> tuple[dict[str, PipelineResult], dict[str, str]]:
    """Diagnose many uploaded files. A bad file is reported, not fatal.

    Returns ({trip label: result}, {file name: error message}).
    """
    pairs, unmatched = pair_uploads(files)
    results, errors = {}, {name: "No driving log with a matching name" for name in unmatched}
    for name, log_bytes, dtc_bytes in pairs:
        try:
            trips = diagnose_upload(log_bytes, name, dtc_file_bytes=dtc_bytes)
        except (ValueError, KeyError) as error:
            errors[name] = short_error(name, str(error))
            continue
        results.update(zip(trip_labels(name, trips), trips))
    return results, errors


def diagnose_samples(keys: list[str], include_dtcs: bool = True) -> dict[str, PipelineResult]:
    """Chosen sample trips, keyed by sample label."""
    by_key = {s.key: s for s in list_samples()}
    return {by_key[k].label: diagnose_sample(k, include_dtcs)[0] for k in keys}
