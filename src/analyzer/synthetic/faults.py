"""Fault definitions: how each fault changes sensor signals and which DTCs it sets.

Every fault is injected into a window of a real VED trip. `intensity` is an
array (one value per row) that ramps from 0 to 1, so the fault develops
gradually, like a real degrading part, instead of appearing all at once.

Fault sizes at full intensity are set near the levels at which a real ECU
would store the DTC, e.g. about +/-25% total fuel trim for P0171 / P0172.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# inject(window, intensity, rng) -> window with modified signal columns
InjectFn = Callable[[pd.DataFrame, np.ndarray, np.random.Generator], pd.DataFrame]


@dataclass(frozen=True)
class Fault:
    name: str
    description: str
    dtc_codes: tuple[str, ...]
    required_signals: tuple[str, ...]
    inject: InjectFn


def _has(window: pd.DataFrame, column: str) -> bool:
    return column in window and window[column].notna().any()


def _low_load_factor(window: pd.DataFrame) -> np.ndarray:
    """1.0 at idle / light load, down to 0.3 at high load (leaks matter most at idle)."""
    load = window["absolute_load_pct"].fillna(30).to_numpy()
    return np.clip(1 - load / 100, 0.3, 1.0)


def _shift_trims(window, amount, rng, banks=(1, 2)):
    """Move short and long term fuel trims by `amount` (array, %)."""
    for bank in banks:
        stft, ltft = f"stft_b{bank}_pct", f"ltft_b{bank}_pct"
        if _has(window, stft):
            window[stft] += 0.4 * amount + rng.normal(0, 1.0, len(window)) * (amount != 0)
        if _has(window, ltft):
            window[ltft] += 0.6 * amount
    return window


def inject_vacuum_leak(window, intensity, rng):
    """Unmetered air enters after the MAF: ECU adds fuel (trims go positive), worst at idle."""
    peak = rng.uniform(25, 35)
    window = _shift_trims(window, intensity * peak * _low_load_factor(window), rng)
    window["maf_gs"] *= 1 - 0.10 * intensity
    return window


def inject_maf_drift(window, intensity, rng):
    """Dirty MAF under-reports airflow; ECU compensates with positive trims."""
    window["maf_gs"] *= 1 - intensity * rng.uniform(0.30, 0.50)
    return _shift_trims(window, intensity * rng.uniform(15, 25), rng)


def inject_rich_injector(window, intensity, rng):
    """Leaking injector adds extra fuel; ECU pulls fuel back (trims go negative)."""
    return _shift_trims(window, -intensity * rng.uniform(20, 30), rng, banks=(1,))


def inject_misfire(window, intensity, rng):
    """Misfiring cylinder makes RPM unsteady, with occasional sharp dips."""
    running = window["engine_rpm"].to_numpy() > 0
    n = len(window)
    jitter = rng.normal(0, rng.uniform(120, 220), n) * intensity
    dips = (rng.random(n) < 0.15 * intensity) * rng.uniform(150, 400, n)
    window["engine_rpm"] += np.where(running, jitter - dips, 0)
    return window


def inject_speed_sensor_failure(window, intensity, rng):
    """Failing speed sensor drops to 0 km/h while the car is clearly moving."""
    moving = window["speed_kmh"].to_numpy() > 10
    dropout = moving & (rng.random(len(window)) < 0.25 * intensity)
    window.loc[dropout, "speed_kmh"] = 0.0
    return window


def inject_hv_battery_degradation(window, intensity, rng):
    """Aged hybrid battery: higher internal resistance -> bigger voltage sag under load."""
    current = window["hv_battery_current_a"].fillna(0).abs().to_numpy()
    resistance = rng.uniform(0.15, 0.30)  # extra ohms at full fault
    offset = rng.uniform(10, 25)           # lower resting voltage (V)
    window["hv_battery_voltage_v"] -= intensity * (resistance * current + offset)
    if _has(window, "hv_battery_soc_pct"):
        window["hv_battery_soc_pct"] -= intensity * rng.uniform(3, 8)
    return window


FAULTS: dict[str, Fault] = {
    f.name: f
    for f in [
        Fault(
            "vacuum_leak",
            "Vacuum leak: extra unmetered air makes the engine run lean",
            ("P0171",),
            ("stft_b1_pct", "absolute_load_pct"),
            inject_vacuum_leak,
        ),
        Fault(
            "maf_drift",
            "Dirty MAF sensor under-reports airflow",
            ("P0101",),
            ("maf_gs", "stft_b1_pct"),
            inject_maf_drift,
        ),
        Fault(
            "rich_injector",
            "Leaking fuel injector makes the engine run rich",
            ("P0172",),
            ("stft_b1_pct",),
            inject_rich_injector,
        ),
        Fault(
            "misfire",
            "Engine misfire causes unstable RPM",
            ("P0300",),
            ("engine_rpm",),
            inject_misfire,
        ),
        Fault(
            "speed_sensor_failure",
            "Vehicle speed sensor intermittently drops out",
            ("P0500",),
            ("speed_kmh", "engine_rpm"),
            inject_speed_sensor_failure,
        ),
        Fault(
            "hv_battery_degradation",
            "Hybrid battery deterioration causes voltage sag under load",
            ("P0A7F",),
            ("hv_battery_voltage_v", "hv_battery_current_a"),
            inject_hv_battery_degradation,
        ),
    ]
}
