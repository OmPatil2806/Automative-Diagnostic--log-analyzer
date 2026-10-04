"""Typed access to Streamlit session state (instead of raw string keys everywhere).

The dashboard keeps every trip diagnosed in this browser session, so the
Overview and Fleet pages can show them together.
"""

import streamlit as st

_TRIPS = "diagnosed_trips"
_CURRENT = "current_diagnosis"


def get_trips() -> dict:
    """All trips diagnosed in this session, keyed by a display name."""
    return st.session_state.setdefault(_TRIPS, {})


def add_trip(name: str, result) -> None:
    get_trips()[name] = result


def clear_trips() -> None:
    st.session_state[_TRIPS] = {}


def set_current(labels: list[str]) -> None:
    """Remember which trips the Diagnose page is showing (survives reruns and page switches)."""
    st.session_state[_CURRENT] = labels


def get_current() -> list[str]:
    """Labels of the trips last diagnosed on the Diagnose page that are still in the session."""
    return [label for label in st.session_state.get(_CURRENT, []) if label in get_trips()]
