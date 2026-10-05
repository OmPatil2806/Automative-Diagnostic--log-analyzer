"""Page header shared by all pages."""

import streamlit as st


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)
