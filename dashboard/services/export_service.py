"""Build downloadable files: one trip in every format, or many trips as a ZIP (no UI)."""

import io
import json
import re
import zipfile
from datetime import datetime

import pandas as pd

from analyzer.pipeline import PipelineResult

TRIP_KEYS = ["vehicle_id", "trip_id"]
FORMATS = {
    "pdf": ("PDF report", "application/pdf"),
    "html": ("HTML report", "text/html"),
    "json": ("JSON data", "application/json"),
    "txt": ("Text report", "text/plain"),
    "csv": ("Scored readings (CSV)", "text/csv"),
}


def safe_name(text: str) -> str:
    """File-system friendly name: 'Vacuum leak / trip 3' -> 'vacuum_leak_trip_3'."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "report"


def base_name(result: PipelineResult, label: str | None = None) -> str:
    report = result.report
    stem = label or report.get("source", "").rsplit(".", 1)[0] or "trip"
    return safe_name(f"{stem}_vehicle{report['vehicle_id']}_trip{report['trip_id']}")


def scored_readings(result: PipelineResult) -> pd.DataFrame:
    """Every reading of the trip with its signals, detector scores and alarm flag."""
    signals = result.trip_logs
    score_cols = [c for c in result.trip_scored.columns if c.startswith("score_")]
    scores = result.trip_scored[TRIP_KEYS + ["time_ms", "anomaly_score", "suspect_group", "alarm", *score_cols]]
    return signals.merge(scores, on=TRIP_KEYS + ["time_ms"], how="left")


def trip_file(result: PipelineResult, fmt: str) -> bytes:
    """One trip's report in one format."""
    if fmt == "pdf":
        return result.pdf()
    if fmt == "html":
        return result.html.encode("utf-8")
    if fmt == "json":
        return json.dumps(result.report, indent=2, default=str).encode("utf-8")
    if fmt == "txt":
        return (result.text + "\n").encode("utf-8")
    if fmt == "csv":
        return scored_readings(result).to_csv(index=False).encode("utf-8")
    raise ValueError(f"Unknown format: {fmt}")


def trip_bundle(result: PipelineResult, label: str | None = None, formats=tuple(FORMATS)) -> bytes:
    """ZIP with one trip's report in every requested format."""
    name = base_name(result, label)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fmt in formats:
            zf.writestr(f"{name}_report.{fmt}" if fmt != "csv" else f"{name}_readings.csv", trip_file(result, fmt))
    return buffer.getvalue()


def fleet_summary(results: dict[str, PipelineResult]) -> pd.DataFrame:
    """One row per trip: health, fault codes, likely cause, early warning."""
    rows = []
    for label, result in results.items():
        r = result.report
        main = r["main_finding"]
        rows.append({
            "trip": label,
            "vehicle_id": r["vehicle_id"],
            "trip_id": r["trip_id"],
            "health_score": r["health"]["score"],
            "status": r["health"]["status"],
            "fault_codes": ", ".join(d["code"] for d in r["dtcs"]),
            "anomalies": len(r["findings"]),
            "early_warnings": sum(f["early_warning"] for f in r["findings"]),
            "likely_cause": main["cause_label"] if main else "",
            "confidence": main["confidence"] if main else "",
            "warning_before_code_s": main["warning_before_dtc_s"] if main else None,
            "duration_min": round(r["trip"]["duration_s"] / 60, 1),
            "distance_km": r["trip"]["distance_km"],
            "source": r.get("source", ""),
        })
    columns = ["trip", "vehicle_id", "trip_id", "health_score", "status", "fault_codes", "anomalies",
               "early_warnings", "likely_cause", "confidence", "warning_before_code_s",
               "duration_min", "distance_km", "source"]
    return pd.DataFrame(rows, columns=columns).sort_values("health_score", kind="stable").reset_index(drop=True)


def fleet_bundle(results: dict[str, PipelineResult], formats=("pdf", "html", "json")) -> bytes:
    """ZIP with a summary CSV plus every trip's reports, one folder per trip."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", fleet_summary(results).to_csv(index=False))
        for label, result in results.items():
            name = base_name(result, label)
            for fmt in formats:
                filename = f"{name}_report.{fmt}" if fmt != "csv" else f"{name}_readings.csv"
                zf.writestr(f"{name}/{filename}", trip_file(result, fmt))
    return buffer.getvalue()


def timestamped(name: str, extension: str) -> str:
    """'fleet_reports' -> 'fleet_reports_2026-10-04_1032.zip'"""
    return f"{name}_{datetime.now():%Y-%m-%d_%H%M}.{extension}"
