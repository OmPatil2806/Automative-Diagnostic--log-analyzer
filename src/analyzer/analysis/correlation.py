"""Correlate anomalies with the DTCs that follow them to suggest root causes.

Three steps per trip:

1. Episodes: consecutive alarm rows are grouped into anomaly episodes
   (short gaps are bridged), each with a suspect subsystem and peak score.
2. Root cause: the signals during an episode are read like a technician
   would, e.g. "fuel trims high + airflow normal -> vacuum leak" but
   "fuel trims high + airflow reading low -> dirty MAF sensor". This uses
   signals only, never the DTC, so it can also explain anomalies that have
   not set a code yet.
3. DTC linking: each DTC is linked to the earliest episode in a matching
   subsystem that started before it (or shortly after). The gap is the early
   warning time. Episodes with no DTC are reported as early warnings.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analyzer.dtc.decoder import decode, load_dtc_table

TRIP_KEYS = ["vehicle_id", "trip_id"]

# Which detector group watches each signal (used to match DTCs to anomalies).
SIGNAL_GROUPS = {
    "engine_rpm": "engine",
    "speed_kmh": "speed",
    "stft_b1_pct": "fuel", "ltft_b1_pct": "fuel", "stft_b2_pct": "fuel", "ltft_b2_pct": "fuel",
    "maf_gs": "air", "absolute_load_pct": "air",
    "hv_battery_voltage_v": "battery", "hv_battery_current_a": "battery", "hv_battery_soc_pct": "battery",
}

# Rule thresholds (in the feature's units).
LEAN_TRIM_PCT = 10.0       # total fuel trim above this = engine running lean
RICH_TRIM_PCT = -10.0      # below this = running rich
MAF_LOW_PCT = -20.0        # airflow this far below expected = MAF under-reading
VOLTAGE_SAG_V = -8.0       # battery voltage this far below expected = sag

DTC_LINK_TOLERANCE_MS = 60_000  # an episode may start up to 60 s after its DTC

CAUSES: dict[str, dict] = {
    "vacuum_leak": {
        "label": "Vacuum leak",
        "checks": ["Inspect intake hoses and vacuum lines", "Check the intake manifold gasket",
                   "Smoke-test the intake for leaks"],
    },
    "maf_drift": {
        "label": "Dirty or faulty MAF sensor",
        "checks": ["Clean or replace the MAF sensor", "Check the air filter", "Check MAF wiring and connector"],
    },
    "rich_injector": {
        "label": "Engine running rich (leaking injector or high fuel pressure)",
        "checks": ["Check fuel injectors for leaks", "Check the fuel pressure regulator",
                   "Check the upstream O2 sensor"],
    },
    "misfire": {
        "label": "Engine misfire",
        "checks": ["Inspect spark plugs", "Test ignition coils", "Check injectors and compression"],
    },
    "speed_sensor_failure": {
        "label": "Faulty vehicle speed sensor",
        "checks": ["Inspect the speed sensor and its wiring", "Check ABS wheel speed sensors"],
    },
    "hv_battery_degradation": {
        "label": "Hybrid battery degradation",
        "checks": ["Run a hybrid battery health test", "Check battery cooling fan and air vents",
                   "Check module voltage balance"],
    },
    "lean_unclear": {
        "label": "Engine running lean (vacuum leak or dirty MAF sensor)",
        "checks": ["Inspect intake hoses and vacuum lines", "Clean or replace the MAF sensor",
                   "Check fuel pressure"],
    },
    "unknown": {
        "label": "Unusual behaviour, cause unclear",
        "checks": ["Monitor the vehicle and re-check if the warning repeats"],
    },
}


@dataclass
class RootCause:
    cause: str
    confidence: str              # high / medium / low
    evidence: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return CAUSES[self.cause]["label"]

    @property
    def checks(self) -> list[str]:
        return CAUSES[self.cause]["checks"]


def dtc_groups(code: str, table: dict | None = None) -> set[str]:
    """Detector groups whose signals a DTC affects (from its related_signals)."""
    info = decode(code, table)
    return {SIGNAL_GROUPS[s] for s in info.related_signals if s in SIGNAL_GROUPS}


def find_episodes(scored: pd.DataFrame, max_gap_s: float = 30, min_duration_s: float = 10) -> pd.DataFrame:
    """Group alarm rows into anomaly episodes.

    Alarms less than `max_gap_s` apart are merged; episodes shorter than
    `min_duration_s` are dropped as noise.
    """
    columns = TRIP_KEYS + ["episode", "start_ms", "end_ms", "duration_s", "suspect_group", "peak_score"]
    alarms = scored[scored["alarm"]].sort_values(TRIP_KEYS + ["time_ms"])
    if alarms.empty:
        return pd.DataFrame(columns=columns)

    gap = alarms.groupby(TRIP_KEYS)["time_ms"].diff()
    new_episode = gap.isna() | (gap > max_gap_s * 1000)
    alarms = alarms.assign(episode=new_episode.groupby([alarms["vehicle_id"], alarms["trip_id"]]).cumsum())

    episodes = alarms.groupby(TRIP_KEYS + ["episode"]).agg(
        start_ms=("time_ms", "min"),
        end_ms=("time_ms", "max"),
        peak_score=("anomaly_score", "max"),
        suspect_group=("suspect_group", lambda s: s.mode().iat[0] if s.notna().any() else None),
    ).reset_index()
    episodes["duration_s"] = (episodes["end_ms"] - episodes["start_ms"]) / 1000
    episodes = episodes[episodes["duration_s"] >= min_duration_s]
    return episodes[columns].reset_index(drop=True)


EVIDENCE_COLUMNS = ["total_trim_rel_median", "maf_residual_median", "rpm_jitter_steady",
                    "speed_zero_while_revving", "speed_jump_rate", "hv_voltage_residual_median"]


def episode_evidence(scored: pd.DataFrame, episode) -> dict[str, float]:
    """Median feature values in the second half of an episode.

    Faults usually grow over time, so the later part shows them most clearly.
    Also adds `maf_change_in_trip_pct`: airflow (per rpm x load) compared with
    the part of the trip before the episode, for vehicles with no baseline.
    """
    trip = scored[(scored["vehicle_id"] == episode.vehicle_id) & (scored["trip_id"] == episode.trip_id)]
    midpoint = (episode.start_ms + episode.end_ms) / 2
    rows = trip[trip["time_ms"].between(midpoint, episode.end_ms)]

    def median(df, column):
        return float(df[column].median()) if column in df and df[column].notna().any() else np.nan

    evidence = {c: median(rows, c) for c in EVIDENCE_COLUMNS}
    before = trip[trip["time_ms"] < episode.start_ms]
    reference = median(before, "maf_ratio_median") if before["maf_ratio_median"].notna().sum() >= 60 else np.nan
    evidence["maf_change_in_trip_pct"] = 100 * (median(rows, "maf_ratio_median") / reference - 1)
    return evidence


def infer_root_cause(evidence: dict[str, float], suspect_group: str | None, peak_score: float = 1.0) -> RootCause:
    """Rule-based diagnosis from signal evidence (no DTC needed)."""
    trim = evidence.get("total_trim_rel_median", np.nan)
    strong = "high" if peak_score >= 2 else "medium"

    # Airflow vs this vehicle's baseline; if there is none, vs earlier in this trip.
    maf = evidence.get("maf_residual_median", np.nan)
    maf_source = "expected for this vehicle"
    if np.isnan(maf):
        maf = evidence.get("maf_change_in_trip_pct", np.nan)
        maf_source = "earlier in this trip"

    if suspect_group in ("fuel", "air"):
        if trim > LEAN_TRIM_PCT:
            lean = f"Fuel trims {trim:+.1f}% above normal: the ECU is adding fuel (lean)"
            if maf < MAF_LOW_PCT:
                return RootCause("maf_drift", strong, [
                    lean, f"Airflow reading {maf:+.0f}% vs {maf_source}: the MAF is under-reporting",
                ])
            if np.isnan(maf):
                return RootCause("lean_unclear", "low", [
                    lean, "No airflow reference available, so a vacuum leak and a dirty MAF are both possible",
                ])
            return RootCause("vacuum_leak", strong, [
                lean,
                f"Airflow reading {maf:+.0f}% vs {maf_source}: close to normal, "
                "so extra air is entering after the MAF",
            ])
        if trim < RICH_TRIM_PCT:
            return RootCause("rich_injector", strong, [
                f"Fuel trims {trim:+.1f}% below normal: the ECU is removing fuel (rich)",
            ])
        if maf < MAF_LOW_PCT:
            return RootCause("maf_drift", "medium", [
                f"Airflow reading {maf:+.0f}% vs {maf_source}",
            ])

    if suspect_group == "engine":
        jitter = evidence.get("rpm_jitter_steady", np.nan)
        return RootCause("misfire", strong, [
            f"RPM unstable at steady speed (jitter {jitter:.0f} rpm)",
        ])

    if suspect_group == "speed":
        drop = evidence.get("speed_zero_while_revving", np.nan)
        return RootCause("speed_sensor_failure", strong, [
            f"Speed reads 0 km/h while moving with the engine loaded ({drop:.0%} of readings)",
        ])

    if suspect_group == "battery":
        sag = evidence.get("hv_voltage_residual_median", np.nan)
        if sag < VOLTAGE_SAG_V:
            return RootCause("hv_battery_degradation", strong, [
                f"Battery voltage {sag:+.0f} V below expected for its charge level and current",
            ])

    return RootCause("unknown", "low", [f"Anomaly in the {suspect_group or 'unknown'} subsystem"])


def link_dtcs(episodes: pd.DataFrame, dtc_events: pd.DataFrame, table: dict | None = None) -> pd.DataFrame:
    """Link each DTC event to the earliest matching episode before it.

    Returns one row per DTC with the linked episode (or NaN) and the early
    warning time in seconds (positive = anomaly seen before the DTC).
    """
    table = load_dtc_table() if table is None else table
    rows = []
    for event in dtc_events.itertuples():
        groups = dtc_groups(event.code, table)
        trip = episodes[(episodes["vehicle_id"] == event.vehicle_id) & (episodes["trip_id"] == event.trip_id)]
        match = trip[
            trip["suspect_group"].isin(groups)
            & (trip["start_ms"] <= event.time_ms + DTC_LINK_TOLERANCE_MS)
        ].sort_values("start_ms")
        linked = match.iloc[0] if len(match) else None
        rows.append({
            "vehicle_id": event.vehicle_id,
            "trip_id": event.trip_id,
            "code": event.code,
            "dtc_time_ms": event.time_ms,
            "episode": linked["episode"] if linked is not None else np.nan,
            "warning_before_dtc_s": (event.time_ms - linked["start_ms"]) / 1000 if linked is not None else np.nan,
        })
    columns = TRIP_KEYS + ["code", "dtc_time_ms", "episode", "warning_before_dtc_s"]
    return pd.DataFrame(rows, columns=columns)
