"""Download buttons for reports. Files are built only when a button is clicked."""

import streamlit as st

from analyzer.pipeline import PipelineResult
from dashboard.services import export_service as es

BUTTONS = [
    ("pdf", "PDF", ":material/picture_as_pdf:", "Printable report with chart"),
    ("html", "HTML", ":material/language:", "Report with interactive charts, opens in any browser"),
    ("json", "JSON", ":material/data_object:", "All report data, for other programs"),
    ("txt", "Text", ":material/description:", "Plain-text report"),
    ("csv", "Readings CSV", ":material/table_view:", "Every reading with its detector scores and alarm flag"),
]


def _file_name(result: PipelineResult, label: str, fmt: str) -> str:
    name = es.base_name(result, label)
    return f"{name}_readings.csv" if fmt == "csv" else f"{name}_report.{fmt}"


def trip_downloads(result: PipelineResult, label: str, key: str) -> None:
    """A ZIP with everything, plus one button per format."""
    st.markdown("**Download this report**")
    st.download_button(
        "Download all formats (ZIP)",
        data=lambda: es.trip_bundle(result, label),
        file_name=f"{es.base_name(result, label)}_reports.zip",
        mime="application/zip",
        icon=":material/folder_zip:",
        type="primary",
        on_click="ignore",
        key=f"{key}_zip",
        help="PDF, HTML, JSON, text report and readings CSV in one file",
    )
    for column, (fmt, name, icon, help_text) in zip(st.columns(len(BUTTONS)), BUTTONS):
        with column:
            st.download_button(
                name,
                data=(lambda f=fmt: es.trip_file(result, f)),
                file_name=_file_name(result, label, fmt),
                mime=es.FORMATS[fmt][1],
                icon=icon,
                on_click="ignore",
                key=f"{key}_{fmt}",
                help=help_text,
                width="stretch",
            )
