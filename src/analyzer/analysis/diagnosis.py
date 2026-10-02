"""Combine anomalies, DTCs and root causes into one diagnosis per trip."""

from dataclasses import dataclass, field

import pandas as pd

from analyzer.analysis.correlation import (
    TRIP_KEYS,
    RootCause,
    episode_evidence,
    find_episodes,
    infer_root_cause,
    link_dtcs,
)
from analyzer.analysis.health_score import health_status, trip_health
from analyzer.dtc.decoder import SEVERITY_LEVELS, decode, load_dtc_table


@dataclass
class Finding:
    """One anomaly episode with its diagnosis."""

    start_ms: int
    end_ms: int
    duration_s: float
    suspect_group: str | None
    peak_score: float
    root_cause: RootCause
    dtc_codes: list[str] = field(default_factory=list)
    warning_before_dtc_s: float | None = None

    @property
    def is_early_warning(self) -> bool:
        """Anomaly seen but no DTC set (yet)."""
        return not self.dtc_codes


@dataclass
class TripDiagnosis:
    vehicle_id: int
    trip_id: int
    health_score: int
    status: str
    dtcs: list[dict] = field(default_factory=list)       # decoded DTCs + link info
    findings: list[Finding] = field(default_factory=list)

    @property
    def main_finding(self) -> Finding | None:
        """Finding linked to the most severe DTC, else the strongest anomaly."""
        if not self.findings:
            return None
        rank = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
        severity = {d["code"]: rank.get(d["severity"], -1) for d in self.dtcs}

        def key(f: Finding):
            return (max((severity.get(c, -1) for c in f.dtc_codes), default=-2), f.peak_score)

        return max(self.findings, key=key)


def diagnose(scored: pd.DataFrame, dtc_events: pd.DataFrame, table: dict | None = None) -> list[TripDiagnosis]:
    """Diagnose every trip in `scored` (output of AnomalyDetector.score)."""
    table = load_dtc_table() if table is None else table
    episodes = find_episodes(scored)
    links = link_dtcs(episodes, dtc_events, table)

    results = []
    for (vehicle_id, trip_id), _ in scored.groupby(TRIP_KEYS):
        trip_eps = episodes[(episodes["vehicle_id"] == vehicle_id) & (episodes["trip_id"] == trip_id)]
        trip_links = links[(links["vehicle_id"] == vehicle_id) & (links["trip_id"] == trip_id)]

        findings = []
        for ep in trip_eps.itertuples():
            linked = trip_links[trip_links["episode"] == ep.episode]
            warning = linked["warning_before_dtc_s"].max() if len(linked) else None
            warning = None if warning is None or pd.isna(warning) else float(warning)
            findings.append(Finding(
                start_ms=int(ep.start_ms),
                end_ms=int(ep.end_ms),
                duration_s=float(ep.duration_s),
                suspect_group=ep.suspect_group,
                peak_score=float(ep.peak_score),
                root_cause=infer_root_cause(episode_evidence(scored, ep), ep.suspect_group, ep.peak_score),
                dtc_codes=linked["code"].tolist(),
                warning_before_dtc_s=warning,
            ))

        dtcs = []
        for link in trip_links.itertuples():
            info = decode(link.code, table)
            dtcs.append({
                "code": info.code,
                "description": info.description,
                "severity": info.severity,
                "advice": info.advice,
                "possible_causes": info.possible_causes,
                "time_ms": int(link.dtc_time_ms) if link.time_known else None,
                "warning_before_dtc_s": None if pd.isna(link.warning_before_dtc_s) else float(link.warning_before_dtc_s),
            })

        severities = [d["severity"] for d in dtcs]
        unexplained = [(f.peak_score, f.duration_s) for f in findings if f.is_early_warning]
        score = trip_health(severities, unexplained)
        results.append(TripDiagnosis(
            vehicle_id=int(vehicle_id),
            trip_id=int(trip_id),
            health_score=score,
            status=health_status(score, severities),
            dtcs=dtcs,
            findings=findings,
        ))
    return results


def to_frame(diagnoses: list[TripDiagnosis]) -> pd.DataFrame:
    """One summary row per trip."""
    rows = []
    for d in diagnoses:
        main = d.main_finding
        rows.append({
            "vehicle_id": d.vehicle_id,
            "trip_id": d.trip_id,
            "health_score": d.health_score,
            "status": d.status,
            "dtc_codes": ",".join(x["code"] for x in d.dtcs),
            "anomaly_episodes": len(d.findings),
            "early_warnings": sum(f.is_early_warning for f in d.findings),
            "likely_cause": main.root_cause.cause if main else None,
            "cause_confidence": main.root_cause.confidence if main else None,
            "warning_before_dtc_s": main.warning_before_dtc_s if main else None,
        })
    return pd.DataFrame(rows)
