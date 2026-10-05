"""Tests for feature engineering and anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from analyzer.detection.anomaly import AnomalyDetector
from analyzer.detection.evaluate import row_metrics, summarize, trip_results
from analyzer.features.engineering import FEATURE_COLUMNS, VehicleBaseline, build_features
from analyzer.synthetic.faults import FAULTS
from analyzer.synthetic.generator import inject_fault
from tests.conftest import make_trip


@pytest.fixture(scope="module")
def normal_logs() -> pd.DataFrame:
    return pd.concat([make_trip(v, t, n=300, seed=v * 100 + t) for v in (1, 2) for t in range(15)],
                     ignore_index=True)


@pytest.fixture(scope="module")
def detector(normal_logs) -> AnomalyDetector:
    return AnomalyDetector(n_estimators=50).fit(normal_logs)


def faulty_trip(name: str, trip_id: int = 500) -> tuple[pd.DataFrame, dict]:
    trip = make_trip(1, trip_id, n=300, seed=trip_id)
    return inject_fault(trip, FAULTS[name], np.random.default_rng(0))


# --- baseline -------------------------------------------------------------

def test_baseline_learns_vehicle_behaviour():
    rng = np.random.default_rng(0)
    n = 1000
    rpm, load = rng.uniform(800, 3000, n), rng.uniform(10, 80, n)
    soc, current = rng.uniform(30, 80, n), rng.normal(0, 40, n)
    logs = pd.DataFrame({
        "vehicle_id": 7, "engine_rpm": rpm, "absolute_load_pct": load,
        "maf_gs": 0.02 * rpm * load / 100,
        "hv_battery_soc_pct": soc, "hv_battery_current_a": current,
        "hv_battery_voltage_v": 200 + 1.5 * soc - 0.1 * current,
        "stft_b1_pct": 3.0, "ltft_b1_pct": -2.0,
    })
    base = VehicleBaseline().fit(logs)
    assert base.maf_k[7] == pytest.approx(0.02)
    assert base.battery_coef[7] == pytest.approx([200, 1.5, -0.1], abs=1e-6)
    res = base.residuals(logs)
    assert res["maf_residual_pct"].abs().max() < 1e-6
    assert res["hv_voltage_residual_v"].abs().max() < 1e-6
    # trims are measured relative to this vehicle's usual level
    assert res["stft_rel"].abs().max() == 0 and res["ltft_rel"].abs().max() == 0


def test_baseline_unknown_vehicle_gives_nan(normal_logs):
    base = VehicleBaseline().fit(normal_logs)
    other = normal_logs.head(50).assign(vehicle_id=999)
    res = base.residuals(other)
    assert res["maf_residual_pct"].isna().all()
    assert res["hv_voltage_residual_v"].isna().all()


# --- features -------------------------------------------------------------

def test_build_features_columns(normal_logs):
    f = build_features(normal_logs, VehicleBaseline().fit(normal_logs))
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert len(f) == len(normal_logs)


def test_speed_dropout_feature():
    base = VehicleBaseline()
    normal = make_trip(1, 1, n=200)
    broken = normal.copy()
    broken.loc[100::3, "speed_kmh"] = 0.0
    f_normal, f_broken = build_features(normal, base), build_features(broken, base)
    assert f_normal["speed_zero_while_revving"].max() == 0
    assert f_broken["speed_zero_while_revving"].max() > 0.2


def test_misfire_raises_rpm_jitter():
    base = VehicleBaseline()
    trip = make_trip(1, 1, n=300).assign(speed_kmh=50.0)   # steady cruise
    noisy = trip.assign(engine_rpm=trip["engine_rpm"] + np.random.default_rng(1).normal(0, 200, 300))
    assert (build_features(noisy, base)["rpm_jitter_steady"].median()
            > 3 * build_features(trip, base)["rpm_jitter_steady"].median())


# --- detector -------------------------------------------------------------

def test_detector_trains_groups(detector):
    assert {"engine", "speed", "fuel", "air", "battery"} <= set(detector.models)


def test_normal_trip_rarely_alarms(detector):
    scored = detector.score(make_trip(1, 900, n=300, seed=900))
    assert scored["alarm"].mean() < 0.05


@pytest.mark.parametrize("fault, group", [
    ("rich_injector", "fuel"),
    ("vacuum_leak", "fuel"),
    ("speed_sensor_failure", "speed"),
    ("hv_battery_degradation", "battery"),
])
def test_fault_raises_alarm_in_right_group(detector, fault, group):
    trip, _ = faulty_trip(fault)
    scored = detector.score(trip)
    full = trip["fault_intensity"].to_numpy() == 1
    assert scored.loc[full, "alarm"].mean() > 0.5
    assert scored.loc[full & scored["alarm"].to_numpy(), "suspect_group"].mode().iat[0] == group


def test_range_score_extrapolates():
    ranges = pd.DataFrame({"low": [-5.0], "median": [0.0], "high": [5.0]}, index=["x"])
    X = pd.DataFrame({"x": [0.0, 5.0, -10.0, 50.0]})
    assert AnomalyDetector._range_score(X, ranges).tolist() == [0.0, 1.0, 2.0, 10.0]


def test_range_score_ignores_zero_width_side():
    ranges = pd.DataFrame({"low": [0.0], "median": [0.0], "high": [0.1]}, index=["rate"])
    X = pd.DataFrame({"rate": [-1.0, 0.3]})
    assert AnomalyDetector._range_score(X, ranges).tolist() == pytest.approx([0.0, 3.0])


def test_min_width_flags_rate_that_is_always_zero_in_training(detector):
    # speed dropouts never happen in normal training data, so the band is just the minimum width
    ranges = detector.ranges["speed"]
    assert ranges.loc["speed_zero_while_revving", "high"] == pytest.approx(0.05)
    X = pd.DataFrame({"speed_jump_rate": [0.0], "speed_zero_while_revving": [0.2]})
    assert AnomalyDetector._range_score(X, ranges)[0] == pytest.approx(4.0)


def test_save_and_load(detector, tmp_path):
    path = tmp_path / "model.joblib"
    detector.save(path)
    trip = make_trip(1, 901, n=200)
    pd.testing.assert_frame_equal(AnomalyDetector.load(path).score(trip), detector.score(trip))


def test_fit_without_enough_data_raises():
    with pytest.raises(ValueError, match="Not enough normal data"):
        AnomalyDetector().fit(make_trip(1, 1, n=100))


# --- evaluation -----------------------------------------------------------

def test_trip_results_and_summary():
    labels = pd.DataFrame({
        "vehicle_id": [1, 1, 1], "trip_id": [1, 2, 3],
        "fault": ["misfire", "misfire", "none"],
        "onset_time_ms": pd.array([10_000, 10_000, pd.NA], dtype="Int64"),
        "dtc_time_ms": pd.array([30_000, 30_000, pd.NA], dtype="Int64"),
    })
    scored = pd.DataFrame({
        "vehicle_id": 1,
        "trip_id": [1, 1, 2, 3],
        "time_ms": [5_000, 20_000, 40_000, 1_000],
        "alarm": [True, True, True, False],
        "suspect_group": ["engine"] * 4,
    })
    trips = trip_results(scored, labels).set_index("trip_id")
    # alarm before onset (5 s) is ignored; first valid alarm 20 s -> 10 s before DTC
    assert trips.loc[1, "first_alarm_ms"] == 20_000
    assert trips.loc[1, "warning_before_dtc_s"] == 10
    assert trips.loc[1, "detected_before_dtc"]
    assert trips.loc[2, "alarm_raised"] and not trips.loc[2, "detected_before_dtc"]
    assert not trips.loc[3, "alarm_raised"]

    summary = summarize(trips.reset_index()).set_index("fault")
    assert summary.loc["misfire", "detection_rate"] == 1.0
    assert summary.loc["misfire", "detected_before_dtc"] == 0.5
    assert summary.loc["none", "false_alarm_rate"] == 0.0


def test_row_metrics():
    logs = pd.DataFrame({
        "vehicle_id": 1, "trip_id": [1, 1, 1, 2],
        "time_ms": [0, 1, 2, 0],
        "fault": ["misfire", "misfire", "misfire", "none"],
        "fault_intensity": [0.5, 1.0, 1.0, 0.0],
    })
    scored = logs[["vehicle_id", "trip_id", "time_ms"]].assign(alarm=[True, True, False, True])
    m = row_metrics(scored, logs)
    # ramp-up row (0.5) excluded: TP=1, FN=1, FP=1
    assert m == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5})


def test_score_methods(detector):
    trip, _ = faulty_trip("rich_injector")
    combined = detector.score(trip)
    forest = detector.score(trip, method="forest")
    range_only = detector.score(trip, method="range")
    for group in ("fuel", "engine"):
        col = f"score_{group}"
        both = pd.concat([forest[col], range_only[col]], axis=1).max(axis=1)
        assert np.allclose(combined[col].dropna(), both.dropna())
    with pytest.raises(ValueError):
        detector.score(trip, method="magic")
