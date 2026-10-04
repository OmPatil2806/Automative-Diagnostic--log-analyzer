"""Diagnosis reports."""

from analyzer.reporting.pdf_report import render_pdf
from analyzer.reporting.report import build_report, render_html, render_text

__all__ = ["build_report", "render_html", "render_pdf", "render_text"]
