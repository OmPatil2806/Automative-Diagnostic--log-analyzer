"""End-to-end pipeline: load -> clean -> score -> decode DTCs -> diagnose -> report.

    from analyzer.pipeline import run
    results = run("data/samples/vacuum_leak.csv", dtc_codes=["P0171"])
    print(results[0].text)
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyzer.analysis.diagnosis import TripDiagnosis, diagnose
from analyzer.config import MODELS_DIR
from analyzer.detection.anomaly import AnomalyDetector
from analyzer.dtc.decoder import is_valid_code, normalize_code, parse_obd_response
from analyzer.ingestion.cleaner import clean_trip_log
from analyzer.ingestion.loader import load_trip_log
from analyzer.reporting.report import build_report, render_html, render_text

DEFAULT_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"
TRIP_KEYS = ["vehicle_id", "trip_id"]
DTC_COLUMNS = ["vehicle_id", "trip_id", "time_ms", "code", "time_known"]


@dataclass
class PipelineResult:
    diagnosis: TripDiagnosis
    report: dict          # JSON-ready
    text: str
    html: str


def collect_dtcs(
    logs: pd.DataFrame,
    dtc_codes: list[str] | None = None,
    obd_response: str | None = None,
    dtc_file: Path | None = None,
) -> pd.DataFrame:
    """Turn the user's DTC inputs into DTC events for every trip in `logs`.

    - `dtc_codes` / `obd_response`: codes read with a scan tool, time unknown.
      They are placed at the end of each trip and get no warning time.
    - `dtc_file`: CSV with `code` and `time_ms` (and optionally vehicle_id, trip_id).
    """
    codes = [normalize_code(c) for c in (dtc_codes or [])]
    if obd_response:
        codes += parse_obd_response(obd_response)
    invalid = [c for c in codes if not is_valid_code(c)]
    if invalid:
        raise ValueError(f"Invalid DTC code(s): {invalid}")

    trip_ends = logs.groupby(TRIP_KEYS)["time_ms"].max().reset_index()
    frames = []
    if codes:
        frames.append(pd.DataFrame([
            {"vehicle_id": t.vehicle_id, "trip_id": t.trip_id, "time_ms": t.time_ms, "code": code, "time_known": False}
            for t in trip_ends.itertuples() for code in dict.fromkeys(codes)
        ]))
    if dtc_file:
        df = pd.read_csv(dtc_file)
        if not {"code", "time_ms"} <= set(df.columns):
            raise ValueError(f"{Path(dtc_file).name} needs columns 'code' and 'time_ms'")
        df["code"] = df["code"].map(normalize_code)
        if "vehicle_id" not in df or "trip_id" not in df:
            # one trip in the log: attach the codes to it
            df = df.drop(columns=[c for c in TRIP_KEYS if c in df]).merge(trip_ends[TRIP_KEYS], how="cross")
        frames.append(df.assign(time_known=True)[DTC_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=DTC_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def run(
    log_path: Path,
    dtc_codes: list[str] | None = None,
    obd_response: str | None = None,
    dtc_file: Path | None = None,
    vehicle_id: int | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    detector: AnomalyDetector | None = None,
    offline_html: bool = False,
) -> list[PipelineResult]:
    """Diagnose every trip in a log file and build its report."""
    if detector is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found. Run `python scripts/train_model.py` first.")
        detector = AnomalyDetector.load(model_path)

    logs = clean_trip_log(load_trip_log(log_path, vehicle_id))
    if logs.empty:
        raise ValueError(f"{Path(log_path).name} has no usable rows after cleaning.")
    events = collect_dtcs(logs, dtc_codes, obd_response, dtc_file)
    scored = detector.score(logs)

    results = []
    for diagnosis in diagnose(scored, events):
        keys = (diagnosis.vehicle_id, diagnosis.trip_id)
        trip_logs = logs[(logs["vehicle_id"] == keys[0]) & (logs["trip_id"] == keys[1])]
        trip_scored = scored[(scored["vehicle_id"] == keys[0]) & (scored["trip_id"] == keys[1])]
        report = build_report(diagnosis, trip_logs, trip_scored, detector, source=Path(log_path).name)
        results.append(PipelineResult(
            diagnosis=diagnosis,
            report=report,
            text=render_text(report),
            html=render_html(report, trip_logs, trip_scored, offline=offline_html),
        ))
    return results
