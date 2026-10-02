"""Tests for fault correlation, root cause and health score."""

import numpy as np
import pandas as pd
import pytest

from analyzer.analysis.correlation import dtc_groups, find_episodes, infer_root_cause, link_dtcs
from analyzer.analysis.diagnosis import diagnose, to_frame
from analyzer.analysis.health_score import (
    ATTENTION,
    CRITICAL,
    GOOD,
    anomaly_penalty,
    health_status,
    trip_health,
    vehicle_health,
)
from analyzer.detection.anomaly import AnomalyDetector
from analyzer.synthetic.faults import FAULTS
from analyzer.synthetic.generator import inject_fault
from tests.conftest import make_trip


def scored_trip(alarm_seconds, trip_id=1, n=200, group="fuel", **features) -> pd.DataFrame:
    """A fake detector output: 1 row per second, alarms at the given seconds."""
    t = np.arange(n)
    df = pd.DataFrame({
        "vehicle_id": 1, "trip_id": trip_id, "time_ms": t * 1000,
        "alarm": np.isin(t, list(alarm_seconds)),
        "anomaly_score": np.where(np.isin(t, list(alarm_seconds)), 2.5, 0.5),
        "suspect_group": group,
        "total_trim_rel_median": 0.0, "maf_residual_median": 0.0, "maf_ratio_median": 0.02,
        "rpm_jitter_steady": 30.0, "speed_zero_while_revving": 0.0, "speed_jump_rate": 0.0,
        "hv_voltage_residual_median": 0.0,
    })
    for column, value in features.items():
        df.loc[df["alarm"], column] = value
    return df


# --- episodes -------------------------------------------------------------

def test_find_episodes_merges_short_gaps_and_drops_noise():
    alarms = list(range(50, 70)) + list(range(80, 100)) + [150, 151]  # gap of 11 s, then a 1 s blip
    episodes = find_episodes(scored_trip(alarms))
    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert (ep.start_ms, ep.end_ms) == (50_000, 99_000)
    assert ep.suspect_group == "fuel"
    assert ep.peak_score == 2.5


def test_find_episodes_without_alarms():
    assert find_episodes(scored_trip([])).empty


# --- root cause rules -----------------------------------------------------

@pytest.mark.parametrize("evidence, group, cause", [
    ({"total_trim_rel_median": 18, "maf_residual_median": 2}, "fuel", "vacuum_leak"),
    ({"total_trim_rel_median": 18, "maf_residual_median": -35}, "fuel", "maf_drift"),
    ({"total_trim_rel_median": 18, "maf_residual_median": np.nan, "maf_change_in_trip_pct": -30}, "fuel", "maf_drift"),
    ({"total_trim_rel_median": 18, "maf_residual_median": np.nan, "maf_change_in_trip_pct": -3}, "fuel", "vacuum_leak"),
    ({"total_trim_rel_median": 18, "maf_residual_median": np.nan}, "fuel", "lean_unclear"),
    ({"total_trim_rel_median": -15}, "fuel", "rich_injector"),
    ({"total_trim_rel_median": 1, "maf_residual_median": -40}, "air", "maf_drift"),
    ({"rpm_jitter_steady": 400}, "engine", "misfire"),
    ({"speed_zero_while_revving": 0.2}, "speed", "speed_sensor_failure"),
    ({"hv_voltage_residual_median": -25}, "battery", "hv_battery_degradation"),
    ({"hv_voltage_residual_median": -2}, "battery", "unknown"),
    ({"total_trim_rel_median": 3}, "fuel", "unknown"),
])
def test_infer_root_cause(evidence, group, cause):
    result = infer_root_cause(evidence, group, peak_score=2.5)
    assert result.cause == cause
    assert result.evidence and result.checks and result.label


def test_root_cause_confidence():
    lean = {"total_trim_rel_median": 18, "maf_residual_median": 2}
    assert infer_root_cause(lean, "fuel", peak_score=3).confidence == "high"
    assert infer_root_cause(lean, "fuel", peak_score=1.2).confidence == "medium"
    assert infer_root_cause({"total_trim_rel_median": 18}, "fuel", 3).confidence == "low"


# --- DTC linking ----------------------------------------------------------

def test_dtc_groups():
    assert dtc_groups("P0171") == {"fuel", "air"}
    assert dtc_groups("P0A7F") == {"battery"}
    assert dtc_groups("P1999") == set()


def test_link_dtcs_matches_group_and_time():
    episodes = pd.DataFrame({
        "vehicle_id": 1, "trip_id": 1, "episode": [1, 2],
        "start_ms": [20_000, 50_000], "end_ms": [40_000, 90_000],
        "duration_s": [20.0, 40.0], "suspect_group": ["speed", "fuel"], "peak_score": [2.0, 3.0],
    })
    events = pd.DataFrame({
        "vehicle_id": 1, "trip_id": 1, "code": ["P0171", "P0A7F", "P0500"],
        "time_ms": [80_000, 80_000, 10_000],
    })
    links = link_dtcs(episodes, events).set_index("code")
    assert links.loc["P0171", "episode"] == 2                     # fuel episode, not the speed one
    assert links.loc["P0171", "warning_before_dtc_s"] == 30       # 80 s - 50 s
    assert np.isnan(links.loc["P0A7F", "episode"])                # no battery anomaly
    assert links.loc["P0500", "warning_before_dtc_s"] == -10      # anomaly 10 s after the DTC (within tolerance)


# --- health score ---------------------------------------------------------

def test_trip_health():
    assert trip_health([], []) == 100
    assert trip_health(["medium"], []) == 75
    assert trip_health(["high", "low"], []) == 55
    assert trip_health(["critical", "critical", "critical"], []) == 0      # floor at 0
    assert trip_health([], [(4.0, 120)]) == 80                             # strong, long anomaly: -20
    assert trip_health([], [(4.0, 120)] * 5) == 70                         # anomaly penalty capped at 30


def test_anomaly_penalty_scales_with_strength_and_duration():
    assert anomaly_penalty(1.0, 60) == 5
    assert anomaly_penalty(4.0, 60) == 20
    assert anomaly_penalty(4.0, 30) == 10


def test_health_status():
    assert health_status(95) == GOOD
    assert health_status(75) == ATTENTION
    assert health_status(40) == CRITICAL
    assert health_status(90, ["critical"]) == CRITICAL


def test_vehicle_health_uses_worst_trip():
    trips = pd.DataFrame({
        "vehicle_id": [1, 1, 2], "trip_id": [10, 11, 20],
        "health_score": [100, 65, 90], "status": [GOOD, ATTENTION, GOOD],
    })
    v = vehicle_health(trips).set_index("vehicle_id")
    assert v.loc[1, "health_score"] == 65 and v.loc[1, "status"] == ATTENTION
    assert v.loc[1, "worst_trip_id"] == 11 and v.loc[1, "trips_with_issues"] == 1
    assert v.loc[2, "status"] == GOOD


# --- diagnosis ------------------------------------------------------------

def test_diagnose_links_finding_and_scores_trip():
    faulty = scored_trip(range(100, 180), trip_id=1, total_trim_rel_median=20, maf_residual_median=1)
    normal = scored_trip([], trip_id=2)
    events = pd.DataFrame({"vehicle_id": [1], "trip_id": [1], "code": ["P0171"], "time_ms": [150_000]})

    results = {d.trip_id: d for d in diagnose(pd.concat([faulty, normal]), events)}
    trip = results[1]
    assert trip.main_finding.root_cause.cause == "vacuum_leak"
    assert trip.main_finding.dtc_codes == ["P0171"]
    assert trip.main_finding.warning_before_dtc_s == 50
    assert trip.dtcs[0]["description"] == "System Too Lean (Bank 1)"
    assert (trip.health_score, trip.status) == (75, ATTENTION)

    assert (results[2].health_score, results[2].status) == (100, GOOD)
    assert results[2].main_finding is None

    frame = to_frame(list(results.values())).set_index("trip_id")
    assert frame.loc[1, "likely_cause"] == "vacuum_leak"
    assert frame.loc[1, "dtc_codes"] == "P0171"


def test_early_warning_without_dtc():
    scored = scored_trip(range(100, 190), group="battery", hv_voltage_residual_median=-25)
    trip = diagnose(scored, pd.DataFrame(columns=["vehicle_id", "trip_id", "code", "time_ms"]))[0]
    finding = trip.main_finding
    assert finding.is_early_warning
    assert finding.root_cause.cause == "hv_battery_degradation"
    assert trip.health_score < 100 and trip.status == GOOD   # one 90 s warning: -12.5 -> 88


def test_end_to_end_with_real_detector():
    normal = pd.concat([make_trip(1, t, n=300, seed=t) for t in range(30)], ignore_index=True)
    detector = AnomalyDetector(n_estimators=50).fit(normal)
    trip, label = inject_fault(make_trip(1, 500, n=300, seed=500), FAULTS["rich_injector"], np.random.default_rng(0))
    events = pd.DataFrame({"vehicle_id": [1], "trip_id": [500], "code": ["P0172"], "time_ms": [label["dtc_time_ms"]]})

    result = diagnose(detector.score(trip), events)[0]
    assert result.main_finding.root_cause.cause == "rich_injector"
    assert result.main_finding.dtc_codes == ["P0172"]
    assert result.status == ATTENTION
