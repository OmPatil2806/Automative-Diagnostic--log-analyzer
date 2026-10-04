"""Page header and placeholder blocks shared by all pages."""

import streamlit as st


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def coming_soon(what: str) -> None:
    st.info(f"{what} is being built and will appear here in an upcoming step.", icon="🛠️")
