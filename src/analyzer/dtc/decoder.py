"""Decode DTCs (e.g. P0301) into description, system, severity and likely causes.

DTC format (SAE J2012):

    P 0 3 0 1
    │ │ │ └─┴── fault number
    │ │ └────── subsystem (for P codes, e.g. 3 = ignition / misfire)
    │ └──────── 0, 2 = generic (SAE)   1 = manufacturer-specific   3 = mixed
    └────────── system: P powertrain, C chassis, B body, U network
"""

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from analyzer.config import REFERENCE_DIR

DTC_TABLE_PATH = REFERENCE_DIR / "dtc_codes.json"

DTC_PATTERN = re.compile(r"^[PCBU][0-3][0-9A-F]{3}$")

SYSTEMS = {"P": "Powertrain", "C": "Chassis", "B": "Body", "U": "Network"}

CODE_TYPES = {
    "0": "Generic (SAE)",
    "1": "Manufacturer-specific",
    "2": "Generic (SAE)",
    "3": "Generic / manufacturer-specific",
}

POWERTRAIN_SUBSYSTEMS = {
    "0": "Fuel and air metering, auxiliary emission controls",
    "1": "Fuel and air metering",
    "2": "Fuel and air metering (injector circuit)",
    "3": "Ignition system or misfire",
    "4": "Auxiliary emission controls",
    "5": "Vehicle speed, idle control and auxiliary inputs",
    "6": "Computer and output circuits",
    "7": "Transmission",
    "8": "Transmission",
    "9": "Transmission",
    "A": "Hybrid propulsion",
    "B": "Hybrid propulsion",
    "C": "Hybrid propulsion",
}

# Ordered from least to most serious.
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
SEVERITY_ADVICE = {
    "low": "Not urgent. Fix at the next service.",
    "medium": "Have it checked soon. Performance or emissions may be affected.",
    "high": "Have it checked as soon as possible. Risk of damage or breakdown.",
    "critical": "Stop driving and get it inspected. Risk of serious damage or safety issue.",
    "unknown": "Not in the reference table. Check the manufacturer's documentation.",
}

# First two bits of an OBD-II DTC byte pair -> system letter.
_OBD_SYSTEM_BITS = "PCBU"


@dataclass
class DTCInfo:
    code: str
    valid: bool
    known: bool = False
    system: str | None = None
    code_type: str | None = None
    subsystem: str | None = None
    description: str = "Unknown code"
    severity: str = "unknown"
    advice: str = SEVERITY_ADVICE["unknown"]
    possible_causes: list[str] = field(default_factory=list)
    related_signals: list[str] = field(default_factory=list)


@lru_cache(maxsize=None)
def load_dtc_table(path: Path = DTC_TABLE_PATH) -> dict[str, dict]:
    """Load the DTC reference table (code -> details)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_code(code: str) -> str:
    """Uppercase and strip spaces/dashes: ' p0301 ' -> 'P0301'."""
    return re.sub(r"[\s-]", "", str(code)).upper()


def is_valid_code(code: str) -> bool:
    """True if `code` has the standard 5-character DTC format."""
    return bool(DTC_PATTERN.match(normalize_code(code)))


def decode(code: str, table: dict[str, dict] | None = None) -> DTCInfo:
    """Decode a single DTC.

    Codes with a valid format but missing from the reference table are still
    decoded structurally (system, code type, subsystem).
    """
    code = normalize_code(code)
    if not DTC_PATTERN.match(code):
        return DTCInfo(code=code, valid=False, description="Invalid DTC format", advice="")

    table = load_dtc_table() if table is None else table
    info = DTCInfo(
        code=code,
        valid=True,
        system=SYSTEMS[code[0]],
        code_type=CODE_TYPES[code[1]],
        subsystem=POWERTRAIN_SUBSYSTEMS.get(code[2]) if code[0] == "P" else None,
    )
    entry = table.get(code)
    if entry:
        info.known = True
        info.description = entry["description"]
        info.severity = entry["severity"]
        info.advice = SEVERITY_ADVICE[entry["severity"]]
        info.possible_causes = list(entry["possible_causes"])
        info.related_signals = list(entry["related_signals"])
    return info


def decode_many(codes: list[str], table: dict[str, dict] | None = None) -> pd.DataFrame:
    """Decode several DTCs into a table, most severe first (unknown, then invalid, last)."""
    rows = [asdict(decode(code, table)) for code in codes]
    df = pd.DataFrame(rows, columns=list(DTCInfo.__dataclass_fields__))
    rank = {level: i for i, level in enumerate(SEVERITY_LEVELS)}
    order = df["severity"].map(rank).fillna(-1).where(df["valid"], -2)
    return df.loc[order.sort_values(ascending=False, kind="stable").index].reset_index(drop=True)


def parse_obd_response(response: str) -> list[str]:
    """Convert a raw OBD-II mode 03 response into DTC strings.

    Each DTC is 2 bytes. The top 2 bits pick the system (P/C/B/U), the next
    2 bits are the first digit, and the remaining 12 bits are three hex digits.
    Example: "43 01 33 00 00" -> ["P0133"]  (leading 43 = mode 03 response)
    Padding pairs of 0000 are ignored.
    """
    hex_str = re.sub(r"\s", "", response).upper()
    if hex_str.startswith("43") and len(hex_str) % 4 == 2:
        hex_str = hex_str[2:]
    if len(hex_str) % 4 or not re.fullmatch(r"[0-9A-F]*", hex_str):
        raise ValueError(f"Not a valid OBD-II DTC response: {response!r}")

    codes = []
    for i in range(0, len(hex_str), 4):
        value = int(hex_str[i:i + 4], 16)
        if value == 0:
            continue
        system = _OBD_SYSTEM_BITS[value >> 14]
        first_digit = (value >> 12) & 0b11
        codes.append(f"{system}{first_digit}{value & 0xFFF:03X}")
    return codes
