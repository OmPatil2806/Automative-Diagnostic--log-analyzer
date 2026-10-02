"""Generate labeled sensor logs with injected faults.

Starts from real, cleaned VED trips. Some trips are kept normal; in the others
one fault is injected from a random onset point to the end of the trip:

    trip:      |------ normal ------|~~~ fault ramps up ~~~|=== full fault ===|
    intensity:          0           0 ─────────────────── 1                   1
                                    ▲ onset               ▲ DTC is set

The DTC is set only once the fault is fully developed, so the signals change
*before* the code appears. That gap is what anomaly detection tries to exploit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analyzer.config import VALID_RANGES
from analyzer.synthetic.faults import FAULTS, Fault

TRIP_KEYS = ["vehicle_id", "trip_id"]
NO_FAULT = "none"


@dataclass
class SyntheticDataset:
    logs: pd.DataFrame        # sensor rows + fault, fault_active, fault_intensity
    dtc_events: pd.DataFrame  # one row per DTC set: vehicle, trip, time, code, fault
    labels: pd.DataFrame      # one row per trip: ground-truth fault, onset and DTC time


def eligible_trips(
    logs: pd.DataFrame, fault: Fault, min_rows: int = 120, min_coverage: float = 0.8
) -> list[tuple[int, int]]:
    """Trips long enough and with enough data in the signals this fault needs."""
    grouped = logs.groupby(TRIP_KEYS)
    ok = grouped.size() >= min_rows
    for signal in fault.required_signals:
        ok &= grouped[signal].apply(lambda s: s.notna().mean()) >= min_coverage
    return list(ok[ok].index)


def _clip_to_valid_ranges(df: pd.DataFrame) -> pd.DataFrame:
    for column, (low, high) in VALID_RANGES.items():
        if column in df:
            df[column] = df[column].clip(low, high)
    return df


def inject_fault(
    trip: pd.DataFrame,
    fault: Fault,
    rng: np.random.Generator,
    onset_range: tuple[float, float] = (0.2, 0.6),
    ramp_fraction: float = 0.4,
) -> tuple[pd.DataFrame, dict]:
    """Inject `fault` into one trip. Returns the modified trip and its label."""
    trip = trip.reset_index(drop=True).copy()
    n = len(trip)
    onset = int(n * rng.uniform(*onset_range))
    ramp = max(1, int((n - onset) * ramp_fraction))
    dtc_index = min(onset + ramp, n - 1)

    intensity = np.zeros(n)
    intensity[onset:] = np.clip((np.arange(n - onset) + 1) / ramp, 0, 1)

    window = trip.iloc[onset:].copy()
    window = fault.inject(window, intensity[onset:], rng)
    trip.iloc[onset:] = _clip_to_valid_ranges(window)

    trip["fault"] = fault.name
    trip["fault_active"] = intensity > 0
    trip["fault_intensity"] = intensity.round(3)

    label = {
        "vehicle_id": int(trip.at[0, "vehicle_id"]),
        "trip_id": int(trip.at[0, "trip_id"]),
        "fault": fault.name,
        "onset_time_ms": int(trip.at[onset, "time_ms"]),
        "dtc_time_ms": int(trip.at[dtc_index, "time_ms"]),
        "dtc_codes": ",".join(fault.dtc_codes),
        "dtc_timestamp": trip.at[dtc_index, "timestamp"] if "timestamp" in trip else pd.NaT,
    }
    return trip, label


def _normal_trip(trip: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    trip = trip.reset_index(drop=True).copy()
    trip["fault"] = NO_FAULT
    trip["fault_active"] = False
    trip["fault_intensity"] = 0.0
    label = {
        "vehicle_id": int(trip.at[0, "vehicle_id"]),
        "trip_id": int(trip.at[0, "trip_id"]),
        "fault": NO_FAULT,
        "onset_time_ms": pd.NA,
        "dtc_time_ms": pd.NA,
        "dtc_codes": "",
        "dtc_timestamp": pd.NaT,
    }
    return trip, label


def generate(
    clean_logs: pd.DataFrame,
    n_faulty: int = 120,
    n_normal: int = 120,
    faults: list[str] | None = None,
    seed: int = 42,
    min_rows: int = 120,
) -> SyntheticDataset:
    """Build a labeled dataset from cleaned VED logs.

    Faulty trips are spread evenly across the fault types (as far as eligible
    trips allow). Every trip is used at most once.
    """
    rng = np.random.default_rng(seed)
    selected = [FAULTS[name] for name in (faults or FAULTS)]
    trips = dict(tuple(clean_logs.groupby(TRIP_KEYS)))
    used: set[tuple[int, int]] = set()

    pools = {f.name: list(rng.permutation(eligible_trips(clean_logs, f, min_rows))) for f in selected}
    out_trips, labels = [], []

    # Round-robin over fault types so the classes stay balanced.
    remaining = n_faulty
    while remaining > 0 and any(pools.values()):
        for fault in selected:
            if remaining == 0:
                break
            pool = pools[fault.name]
            while pool and tuple(pool[-1]) in used:
                pool.pop()
            if not pool:
                continue
            key = tuple(pool.pop())
            used.add(key)
            trip, label = inject_fault(trips[key], fault, rng)
            out_trips.append(trip)
            labels.append(label)
            remaining -= 1

    long_enough = [k for k, t in trips.items() if len(t) >= min_rows and k not in used]
    for i in rng.permutation(len(long_enough))[:n_normal]:
        key = long_enough[i]
        used.add(key)
        trip, label = _normal_trip(trips[key])
        out_trips.append(trip)
        labels.append(label)

    if not out_trips:
        raise ValueError("No eligible trips found. Load more VED weeks.")

    logs = pd.concat(out_trips, ignore_index=True).sort_values(TRIP_KEYS + ["time_ms"])
    labels_df = pd.DataFrame(labels).sort_values(TRIP_KEYS).reset_index(drop=True)
    labels_df = labels_df.astype({"onset_time_ms": "Int64", "dtc_time_ms": "Int64"})
    return SyntheticDataset(
        logs=logs.reset_index(drop=True),
        dtc_events=_dtc_events(labels_df),
        labels=labels_df,
    )


def _dtc_events(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per DTC set, like the fault memory a scan tool would read."""
    faulty = labels[labels["fault"] != NO_FAULT]
    rows = [
        {
            "vehicle_id": row.vehicle_id,
            "trip_id": row.trip_id,
            "time_ms": row.dtc_time_ms,
            "timestamp": row.dtc_timestamp,
            "code": code,
            "fault": row.fault,
        }
        for row in faulty.itertuples()
        for code in row.dtc_codes.split(",")
    ]
    columns = ["vehicle_id", "trip_id", "time_ms", "timestamp", "code", "fault"]
    return pd.DataFrame(rows, columns=columns)
