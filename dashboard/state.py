"""Typed access to Streamlit session state (instead of raw string keys everywhere).

The dashboard keeps every trip diagnosed in this browser session, so the
Overview and Fleet pages can show them together.
"""

import streamlit as st

_TRIPS = "diagnosed_trips"


def get_trips() -> dict:
    """All trips diagnosed in this session, keyed by a display name."""
    return st.session_state.setdefault(_TRIPS, {})


def add_trip(name: str, result) -> None:
    get_trips()[name] = result


def clear_trips() -> None:
    st.session_state[_TRIPS] = {}
