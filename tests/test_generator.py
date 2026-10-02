"""Tests for synthetic fault generation."""

import numpy as np
import pandas as pd
import pytest

from analyzer.config import VALID_RANGES
from analyzer.synthetic.faults import FAULTS
from analyzer.synthetic.generator import NO_FAULT, eligible_trips, generate, inject_fault
from tests.conftest import make_trip


@pytest.fixture
def clean_logs() -> pd.DataFrame:
    trips = [make_trip(1, i, seed=i) for i in range(30)]
    trips.append(make_trip(2, 99, n=50))                       # too short
    no_trims = make_trip(3, 1).assign(stft_b1_pct=np.nan, ltft_b1_pct=np.nan)
    trips.append(no_trims)                                      # missing fuel trims
    return pd.concat(trips, ignore_index=True)


# --- single fault injection -----------------------------------------------

# fault -> (signal, expected direction of change once the fault is fully developed)
EXPECTED = {
    "vacuum_leak": ("ltft_b1_pct", +1),
    "maf_drift": ("maf_gs", -1),
    "rich_injector": ("ltft_b1_pct", -1),
    "speed_sensor_failure": ("speed_kmh", -1),
    "hv_battery_degradation": ("hv_battery_voltage_v", -1),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_fault_shifts_expected_signal(name):
    trip = make_trip(1, 1)
    faulty, _ = inject_fault(trip, FAULTS[name], np.random.default_rng(0))
    signal, direction = EXPECTED[name]
    full = faulty["fault_intensity"] == 1
    change = faulty.loc[full, signal].mean() - trip.loc[full, signal].mean()
    assert np.sign(change) == direction


def test_misfire_makes_rpm_unstable():
    trip = make_trip(1, 1)
    faulty, _ = inject_fault(trip, FAULTS["misfire"], np.random.default_rng(0))
    full = faulty["fault_intensity"] == 1
    assert faulty.loc[full, "engine_rpm"].diff().std() > 2 * trip.loc[full, "engine_rpm"].diff().std()


@pytest.mark.parametrize("name", list(FAULTS))
def test_signals_unchanged_before_onset(name):
    trip = make_trip(1, 1)
    faulty, _ = inject_fault(trip, FAULTS[name], np.random.default_rng(0))
    before = ~faulty["fault_active"]
    pd.testing.assert_frame_equal(faulty.loc[before, trip.columns], trip.loc[before])


@pytest.mark.parametrize("name", list(FAULTS))
def test_values_stay_in_valid_ranges(name):
    faulty, _ = inject_fault(make_trip(1, 1), FAULTS[name], np.random.default_rng(0))
    for column, (low, high) in VALID_RANGES.items():
        if column in faulty:
            assert faulty[column].dropna().between(low, high).all(), column


def test_intensity_ramps_and_dtc_after_onset():
    faulty, label = inject_fault(make_trip(1, 1), FAULTS["vacuum_leak"], np.random.default_rng(0))
    intensity = faulty["fault_intensity"]
    assert intensity.is_monotonic_increasing
    assert intensity.iloc[0] == 0 and intensity.iloc[-1] == 1
    assert label["dtc_time_ms"] > label["onset_time_ms"]
    # the DTC is set when the fault is fully developed
    at_dtc = faulty.loc[faulty["time_ms"] == label["dtc_time_ms"], "fault_intensity"].item()
    assert at_dtc == 1
    assert label["dtc_codes"] == "P0171"


# --- dataset generation ---------------------------------------------------

def test_eligible_trips_filters_short_and_missing(clean_logs):
    trips = eligible_trips(clean_logs, FAULTS["vacuum_leak"])
    assert (2, 99) not in trips      # too short
    assert (3, 1) not in trips       # no fuel trims
    assert len(trips) == 30


def test_generate_balanced_and_unique(clean_logs):
    data = generate(clean_logs, n_faulty=12, n_normal=10, seed=1)
    counts = data.labels["fault"].value_counts()
    assert counts[NO_FAULT] == 10
    assert all(counts[name] == 2 for name in FAULTS)
    assert not data.labels.duplicated(["vehicle_id", "trip_id"]).any()
    assert len(data.dtc_events) == 12


def test_normal_trips_untouched(clean_logs):
    data = generate(clean_logs, n_faulty=6, n_normal=5, seed=1)
    normal = data.logs[data.logs["fault"] == NO_FAULT]
    assert not normal["fault_active"].any()
    merged = normal.merge(clean_logs, on=["vehicle_id", "trip_id", "time_ms"], suffixes=("", "_orig"))
    assert len(merged) == len(normal)
    assert np.allclose(merged["engine_rpm"], merged["engine_rpm_orig"])


def test_dtc_events_match_labels(clean_logs):
    data = generate(clean_logs, n_faulty=6, n_normal=0, seed=1)
    events = data.dtc_events.merge(data.labels, on=["vehicle_id", "trip_id"])
    assert (events["time_ms"] == events["dtc_time_ms"]).all()
    assert (events["fault_x"] == events["fault_y"]).all()


def test_generate_is_reproducible(clean_logs):
    a = generate(clean_logs, n_faulty=6, n_normal=3, seed=5)
    b = generate(clean_logs, n_faulty=6, n_normal=3, seed=5)
    pd.testing.assert_frame_equal(a.logs, b.logs)


def test_generate_subset_of_faults(clean_logs):
    data = generate(clean_logs, n_faulty=4, n_normal=0, faults=["misfire"], seed=1)
    assert set(data.labels["fault"]) == {"misfire"}


def test_generate_without_eligible_trips_raises():
    with pytest.raises(ValueError, match="No eligible trips"):
        generate(make_trip(1, 1, n=50), n_faulty=2, n_normal=2)
