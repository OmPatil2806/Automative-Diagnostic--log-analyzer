"""Streamlit dashboard entry point.

Run from the project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Automotive Diagnostic Log Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation({
    "Diagnosis": [
        st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(PAGES / "diagnose.py", title="Diagnose a trip", icon=":material/troubleshoot:"),
        st.Page(PAGES / "fleet_reports.py", title="Fleet reports", icon=":material/folder_zip:"),
    ],
    "Analysis": [
        st.Page(PAGES / "signal_explorer.py", title="Signal explorer", icon=":material/monitoring:"),
        st.Page(PAGES / "model_performance.py", title="Model performance", icon=":material/insights:"),
    ],
    "Help": [
        st.Page(PAGES / "about.py", title="About", icon=":material/info:"),
    ],
})

with st.sidebar:
    st.markdown("### 🚗 Diagnostic Log Analyzer")
    st.caption("Finds vehicle faults in OBD-II driving logs, explains the likely cause, and scores vehicle health.")

navigation.run()
