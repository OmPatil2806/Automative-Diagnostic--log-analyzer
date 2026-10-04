"""Stat tiles and status badges."""

import html

import streamlit as st

from dashboard.theme import STATUS


def status_badge(status: str) -> str:
    """HTML badge: colored dot + text label (status is never shown by color alone)."""
    meta = STATUS.get(status, {"color": "#7a7974", "label": str(status).title()})
    return (f'<span class="dla-badge"><span class="dla-dot" style="background:{meta["color"]}"></span>'
            f'{html.escape(meta["label"])}</span>')


def kpi_row(items: list[tuple[str, str | int | float, str | None]]) -> None:
    """A row of bordered stat tiles: (label, value, help text)."""
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        with column.container(border=True):
            st.metric(label, value, help=help_text)


def status_counts(statuses: list[str]) -> list[tuple[str, int, None]]:
    """Tiles for how many trips are in each health status."""
    return [(STATUS[s]["label"], sum(x == s for x in statuses), None) for s in STATUS]
