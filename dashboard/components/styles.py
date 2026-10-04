"""Small CSS layer for the HTML pieces Streamlit has no native widget for (badges, chips, score)."""

import streamlit as st

from dashboard.theme import BORDER, CARD, MUTED, TEXT, TEXT_2

CSS = f"""
<style>
.dla-badge {{ display:inline-flex; align-items:center; gap:8px; padding:3px 12px 3px 9px; border-radius:999px;
  border:1px solid {BORDER}; background:{CARD}; font-weight:600; font-size:0.9rem; color:{TEXT}; white-space:nowrap; }}
.dla-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; }}
.dla-score {{ font-size:3.2rem; font-weight:700; line-height:1; color:{TEXT}; }}
.dla-score small {{ font-size:1rem; font-weight:400; color:{TEXT_2}; }}
.dla-chip {{ display:inline-block; font-size:0.78rem; font-weight:600; color:{TEXT_2}; border:1px solid {BORDER};
  border-radius:6px; padding:1px 7px; margin:0 6px 4px 0; white-space:nowrap; }}
.dla-muted {{ color:{MUTED}; font-size:0.85rem; }}
.dla-table {{ width:100%; border-collapse:collapse; font-size:0.92rem; border:none !important; }}
.dla-table tr {{ border:none !important; background:transparent !important; }}
.dla-table th {{ text-align:left; color:{TEXT_2}; font-weight:600; font-size:0.8rem; padding:6px 8px !important;
  border:none !important; border-bottom:1px solid {MUTED} !important; background:transparent !important; }}
.dla-table td {{ padding:8px !important; border:none !important; border-bottom:1px solid {BORDER} !important;
  vertical-align:top; color:{TEXT}; }}
.dla-code {{ font-weight:700; font-family:ui-monospace, Menlo, Consolas, monospace; color:{TEXT}; }}
.dla-section {{ font-size:1.05rem; font-weight:700; color:{TEXT}; margin:1.2rem 0 0.5rem; }}
</style>
"""


def inject_styles() -> None:
    """Add the CSS to the page (cheap; Streamlit dedupes identical markdown)."""
    st.markdown(CSS, unsafe_allow_html=True)
