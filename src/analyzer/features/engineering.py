"""Build ML features from sensor signals (rolling stats, rates of change, residuals).

Two kinds of features:

1. Vehicle-relative residuals. Normal values differ between vehicles (engine
   size, battery pack, each engine's usual fuel trim), so a `VehicleBaseline`
   learns each vehicle's normal behaviour from fault-free trips and we measure
   the difference from it:
     - airflow:     maf ≈ k * rpm * load        -> maf_residual_pct
     - battery:     V ≈ a + b * SOC + c * I     -> hv_voltage_residual_v
     - fuel trims:  usual STFT / LTFT level     -> stft_rel, ltft_rel

2. Rolling features over a window within each trip (e.g. RPM jitter at steady
   speed, impossible speed jumps, median fuel trim deviation). Medians are used
   for noisy signals so single spikes do not dominate.

Only "deviation" features are used (no raw speed or RPM level), so ordinary but
less common driving, like a highway run, is not mistaken for a fault.

Features are grouped by subsystem so each group can get its own detector, and
an alarm can say which subsystem looks wrong.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRIP_KEYS = ["vehicle_id", "trip_id"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "engine": ["rpm_jitter_steady", "rpm_dip_rate"],
    "speed": ["speed_jump_rate", "speed_zero_while_revving"],
    "fuel": ["stft_rel_median", "ltft_rel_median", "total_trim_rel_median"],
    "air": ["maf_residual_median"],
    "battery": ["hv_voltage_residual_median", "hv_voltage_residual_min"],
}
FEATURE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]

# Smallest "normal band" half-width per feature, in the feature's own units.
# Keeps the range score meaningful when normal data barely varies, e.g. a
# dropout rate that is always 0 (any dropout is then clearly abnormal).
FEATURE_MIN_WIDTH: dict[str, float] = {
    "rpm_jitter_steady": 20.0,          # rpm
    "rpm_dip_rate": 0.05,               # share of samples
    "speed_jump_rate": 0.05,
    "speed_zero_while_revving": 0.05,
    "stft_rel_median": 2.0,             # % trim
    "ltft_rel_median": 2.0,
    "total_trim_rel_median": 3.0,
    "maf_residual_median": 5.0,         # % airflow
    "hv_voltage_residual_median": 3.0,  # volts
    "hv_voltage_residual_min": 3.0,
}

MIN_BASELINE_ROWS = 200
MAX_ACCEL_KMH_PER_S = 15.0  # a car cannot change speed faster than this


@dataclass
class VehicleBaseline:
    """Per-vehicle normal behaviour learned from fault-free data."""

    maf_k: dict[int, float] = field(default_factory=dict)
    battery_coef: dict[int, np.ndarray] = field(default_factory=dict)
    stft_offset: dict[int, float] = field(default_factory=dict)
    ltft_offset: dict[int, float] = field(default_factory=dict)

    def fit(self, logs: pd.DataFrame) -> "VehicleBaseline":
        for vehicle_id, df in logs.groupby("vehicle_id"):
            air = df[(df["engine_rpm"] > 500) & (df["absolute_load_pct"] > 5) & (df["maf_gs"] > 0)]
            if len(air) >= MIN_BASELINE_ROWS:
                ratio = air["maf_gs"] / (air["engine_rpm"] * air["absolute_load_pct"] / 100)
                self.maf_k[vehicle_id] = float(ratio.median())

            cols = ["hv_battery_voltage_v", "hv_battery_soc_pct", "hv_battery_current_a"]
            batt = df[cols].dropna()
            if len(batt) >= MIN_BASELINE_ROWS:
                X = np.column_stack([np.ones(len(batt)), batt[cols[1]], batt[cols[2]]])
                coef, *_ = np.linalg.lstsq(X, batt[cols[0]].to_numpy(), rcond=None)
                self.battery_coef[vehicle_id] = coef

            for column, offsets in [("stft_b1_pct", self.stft_offset), ("ltft_b1_pct", self.ltft_offset)]:
                if df[column].notna().sum() >= MIN_BASELINE_ROWS:
                    offsets[vehicle_id] = float(df[column].median())
        return self

    def residuals(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Differences from each vehicle's normal behaviour.

        Unknown vehicles get NaN for airflow and battery; fuel trims fall back
        to an offset of 0 (a healthy engine's trims sit near 0).
        """
        out = pd.DataFrame(index=logs.index)
        k = logs["vehicle_id"].map(self.maf_k)
        expected_maf = k * logs["engine_rpm"] * logs["absolute_load_pct"] / 100
        valid = (logs["engine_rpm"] > 500) & (logs["absolute_load_pct"] > 5)
        out["maf_residual_pct"] = (100 * (logs["maf_gs"] / expected_maf - 1)).where(valid)

        coef = logs["vehicle_id"].map(self.battery_coef)
        has = coef.notna()
        expected_v = pd.Series(np.nan, index=logs.index)
        if has.any():
            c = np.vstack(coef[has].to_numpy())
            expected_v[has] = (
                c[:, 0]
                + c[:, 1] * logs.loc[has, "hv_battery_soc_pct"]
                + c[:, 2] * logs.loc[has, "hv_battery_current_a"]
            )
        out["hv_voltage_residual_v"] = logs["hv_battery_voltage_v"] - expected_v

        out["stft_rel"] = logs["stft_b1_pct"] - logs["vehicle_id"].map(self.stft_offset).fillna(0)
        out["ltft_rel"] = logs["ltft_b1_pct"] - logs["vehicle_id"].map(self.ltft_offset).fillna(0)
        return out


def _per_trip(logs: pd.DataFrame, values: pd.Series, func: str, window: int, min_periods: int) -> pd.Series:
    """Rolling `func` of `values` within each trip (rows must be sorted by trip and time)."""
    grouped = values.groupby([logs["vehicle_id"], logs["trip_id"]])
    rolled = getattr(grouped.rolling(window, min_periods=min_periods), func)()
    return rolled.reset_index(level=[0, 1], drop=True).reindex(logs.index)


def build_features(
    logs: pd.DataFrame, baseline: VehicleBaseline, window: int = 60, min_periods: int = 15
) -> pd.DataFrame:
    """Compute one feature row per log row (window = number of samples, ~1 s each)."""
    logs = logs.sort_values(TRIP_KEYS + ["time_ms"])
    trip = [logs["vehicle_id"], logs["trip_id"]]

    def roll(values, func="mean"):
        return _per_trip(logs, values, func, window, min_periods)

    dt_s = logs.groupby(trip)["time_ms"].diff().div(1000).clip(lower=0.1)
    rpm, speed = logs["engine_rpm"], logs["speed_kmh"]
    rpm_step = rpm.groupby(trip).diff()
    speed_step = speed.groupby(trip).diff()
    prev_speed = speed.groupby(trip).shift()

    f = pd.DataFrame(index=logs.index)
    f[TRIP_KEYS + ["time_ms"]] = logs[TRIP_KEYS + ["time_ms"]]

    # engine: RPM jitter while speed is steady (normal RPM changes come from
    # speeding up / slowing down; a misfire makes RPM jump at constant speed)
    steady = (speed_step.abs() <= 1) & (speed_step.groupby(trip).shift().abs() <= 1) & (rpm > 0)
    rpm_jerk = rpm_step.groupby(trip).diff().abs()
    f["rpm_jitter_steady"] = roll(rpm_jerk.where(steady), "median")
    f["rpm_dip_rate"] = roll(((rpm_step < -150) & steady).astype(float).where(rpm_step.notna()))

    # speed: physically impossible jumps, or 0 km/h right after moving with engine loaded
    accel = (speed_step / dt_s).abs()
    f["speed_jump_rate"] = roll((accel > MAX_ACCEL_KMH_PER_S).astype(float).where(accel.notna()))
    zero_while_revving = (speed == 0) & (prev_speed > 10) & (rpm > 1000)
    f["speed_zero_while_revving"] = roll(zero_while_revving.astype(float).where(prev_speed.notna()))

    # differences from this vehicle's normal behaviour
    res = baseline.residuals(logs)

    # fuel: trims away from the vehicle's usual level = ECU correcting lean / rich
    f["stft_rel_median"] = roll(res["stft_rel"], "median")
    f["ltft_rel_median"] = roll(res["ltft_rel"], "median")
    f["total_trim_rel_median"] = roll(res["stft_rel"] + res["ltft_rel"], "median")

    # air and battery
    f["maf_residual_median"] = roll(res["maf_residual_pct"], "median")
    f["hv_voltage_residual_median"] = roll(res["hv_voltage_residual_v"], "median")
    f["hv_voltage_residual_min"] = roll(res["hv_voltage_residual_v"], "min")

    return f
