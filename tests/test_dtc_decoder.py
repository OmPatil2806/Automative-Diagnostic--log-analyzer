"""Tests for DTC decoding."""

import pytest

from analyzer.config import VED_COLUMNS
from analyzer.dtc.decoder import (
    SEVERITY_LEVELS,
    decode,
    decode_many,
    is_valid_code,
    load_dtc_table,
    normalize_code,
    parse_obd_response,
)


# --- reference table ------------------------------------------------------

def test_table_entries_are_well_formed():
    table = load_dtc_table()
    signals = set(VED_COLUMNS.values())
    assert len(table) >= 50
    for code, entry in table.items():
        assert is_valid_code(code), code
        assert entry["description"], code
        assert entry["severity"] in SEVERITY_LEVELS, code
        assert entry["possible_causes"], code
        assert set(entry["related_signals"]) <= signals, code


# --- format ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [(" p0301 ", "P0301"), ("P-0A80", "P0A80"), ("u0100", "U0100")])
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


@pytest.mark.parametrize("code", ["P0301", "C0035", "B0001", "U0100", "P0A80", "P1234"])
def test_valid_codes(code):
    assert is_valid_code(code)


@pytest.mark.parametrize("code", ["", "P030", "P03011", "X0301", "P4301", "P03G1"])
def test_invalid_codes(code):
    assert not is_valid_code(code)


# --- decode ---------------------------------------------------------------

def test_decode_known_code():
    info = decode("p0301")
    assert info.valid and info.known
    assert info.code == "P0301"
    assert info.system == "Powertrain"
    assert info.code_type == "Generic (SAE)"
    assert info.subsystem == "Ignition system or misfire"
    assert info.description == "Cylinder 1 Misfire Detected"
    assert info.severity == "high"
    assert "engine_rpm" in info.related_signals
    assert info.possible_causes


def test_decode_unknown_but_valid_code():
    info = decode("P1456")
    assert info.valid and not info.known
    assert info.code_type == "Manufacturer-specific"
    assert info.subsystem == "Auxiliary emission controls"
    assert info.severity == "unknown"


def test_decode_non_powertrain_has_no_subsystem():
    info = decode("U0100")
    assert info.system == "Network"
    assert info.subsystem is None
    assert info.severity == "critical"


def test_decode_invalid_code():
    info = decode("hello")
    assert not info.valid
    assert info.description == "Invalid DTC format"


def test_decode_with_custom_table():
    table = {"P0001": {"description": "Test", "severity": "low", "possible_causes": ["x"], "related_signals": []}}
    assert decode("P0001", table).description == "Test"
    assert not decode("P0301", table).known


def test_decode_many_sorted_by_severity():
    df = decode_many(["P0442", "BAD", "P0217", "P1999", "P0301"])
    assert df["code"].tolist() == ["P0217", "P0301", "P0442", "P1999", "BAD"]
    assert df["severity"].tolist()[:3] == ["critical", "high", "low"]


# --- raw OBD-II responses -------------------------------------------------

@pytest.mark.parametrize("response, expected", [
    ("43 01 33 00 00 00 00", ["P0133"]),
    ("0301 0420", ["P0301", "P0420"]),
    ("4100", ["C0100"]),
    ("8001", ["B0001"]),
    ("C100", ["U0100"]),
    ("0A80", ["P0A80"]),
    ("", []),
])
def test_parse_obd_response(response, expected):
    assert parse_obd_response(response) == expected


def test_parse_obd_response_invalid():
    with pytest.raises(ValueError):
        parse_obd_response("01Z3")
