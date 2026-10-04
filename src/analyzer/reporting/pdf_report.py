"""PDF diagnosis report (ReportLab).

`render_pdf(report, trip_logs, trip_scored)` returns the PDF as bytes, with the
same content as the HTML report: header, health score, fault codes, anomalies
with likely causes, recommended checks, a signal chart and the checks that ran.

Uses the DejaVu Sans font shipped with matplotlib, so the status symbols
(✓ ⚠ ✗) and other characters render the same as in the text and HTML reports.
The chart is drawn with matplotlib (no browser or extra tools needed).
"""

import io
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from analyzer.reporting.charts import main_signal
from analyzer.reporting.report import STATUS_ICON, STATUS_TEXT, fmt_time

# --- colors (same tokens as the HTML report) --------------------------------
TEXT = colors.HexColor("#0b0b0b")
TEXT_2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#7a7974")
BORDER = colors.HexColor("#e2e1dc")
PANEL = colors.HexColor("#f6f5f2")
BLUE = "#2a78d6"
SHADE = "#d03b3b"
STATUS_COLOR = {"good": "#0ca30c", "needs attention": "#fab219", "critical": "#d03b3b"}
SEVERITY_COLOR = {"low": "#fab219", "medium": "#ec835a", "high": "#d03b3b", "critical": "#d03b3b", "unknown": "#7a7974"}

# --- fonts ----------------------------------------------------------------
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT, FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"


def _register_fonts() -> None:
    if FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT, str(_FONT_DIR / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
        # so <b> in paragraphs switches to the bold font
        for bold, italic, face in [(0, 0, FONT), (1, 0, FONT_BOLD), (0, 1, FONT), (1, 1, FONT_BOLD)]:
            addMapping(FONT, bold, italic, face)


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName=FONT, fontSize=9.5, leading=13.5, textColor=TEXT, alignment=TA_LEFT)
    return {
        "base": base,
        "title": ParagraphStyle("title", parent=base, fontName=FONT_BOLD, fontSize=18, leading=22),
        "meta": ParagraphStyle("meta", parent=base, textColor=TEXT_2, fontSize=9),
        "h2": ParagraphStyle("h2", parent=base, fontName=FONT_BOLD, fontSize=12, leading=16, spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=base, fontName=FONT_BOLD, fontSize=10.5, leading=14),
        "small": ParagraphStyle("small", parent=base, fontSize=8.5, leading=12, textColor=TEXT_2),
        "muted": ParagraphStyle("muted", parent=base, fontSize=8.5, leading=12, textColor=MUTED),
        "score": ParagraphStyle("score", parent=base, fontName=FONT_BOLD, fontSize=30, leading=34),
        "cell": ParagraphStyle("cell", parent=base, fontSize=9, leading=12),
        "cell_head": ParagraphStyle("cell_head", parent=base, fontName=FONT_BOLD, fontSize=8.5, leading=11, textColor=TEXT_2),
    }


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dot(color: str) -> str:
    return f'<font color="{color}">●</font>'


# --- chart ----------------------------------------------------------------

def signal_chart_png(report: dict, trip_logs: pd.DataFrame, trip_scored: pd.DataFrame, width_in: float = 7.2) -> bytes:
    """Speed / key signal / anomaly score panels as a PNG (matplotlib, no pyplot state)."""
    main = report.get("main_finding")
    signal, signal_title = main_signal(main["subsystem_key"] if main else None, trip_logs, trip_scored)
    panels = [
        (trip_logs.set_index("time_ms")["speed_kmh"], "Speed (km/h)"),
        (signal, signal_title),
        (trip_scored.set_index("time_ms")["anomaly_score"], "Anomaly score (1 = normal limit)"),
    ]
    t0 = trip_logs["time_ms"].min()

    fig = Figure(figsize=(width_in, 4.6), dpi=200)
    axes = fig.subplots(3, 1, sharex=True)
    for ax, (values, title) in zip(axes, panels):
        ax.plot((values.index - t0) / 60_000, values.to_numpy(), color=BLUE, lw=1.1)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color="#0b0b0b", pad=3)
        ax.tick_params(labelsize=7, colors="#52514e", length=0)
        ax.grid(color="#e8e7e3", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#e2e1dc")
        for f in report["findings"]:
            ax.axvspan(f["start_s"] / 60, f["end_s"] / 60, color=SHADE, alpha=0.12, lw=0)
        for d in report["dtcs"]:
            if d["time_s"] is not None:
                ax.axvline(d["time_s"] / 60, color="#0b0b0b", lw=1, ls=":")
    axes[2].axhline(1, color="#52514e", lw=0.8, ls="--")
    for d in report["dtcs"]:
        if d["time_s"] is not None:
            axes[2].annotate(f"{d['code']} set ", (d["time_s"] / 60, 0.95), xycoords=("data", "axes fraction"),
                             ha="right", va="top", fontsize=7, color="#0b0b0b")
    axes[2].set_xlabel("time into trip (min)", fontsize=7.5, color="#52514e")
    fig.tight_layout(h_pad=0.8)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


# --- document -------------------------------------------------------------

def _table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED))
    table.setStyle(TableStyle(style))
    return table


def _health_block(report: dict, s: dict, width: float) -> Table:
    h = report["health"]
    status = h["status"]
    badge = f'{_dot(STATUS_COLOR[status])} <b>{STATUS_ICON[status]} {STATUS_TEXT[status]}</b>'
    left = [Paragraph(f'{h["score"]}<font size="11" color="#52514e"> / 100</font>', s["score"]),
            Spacer(1, 4), Paragraph(badge, s["base"])]
    block = Table([[left, Paragraph(_esc(h["summary"]), s["base"])]], colWidths=[62 * mm, width - 62 * mm])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return block


def _footer(source: str, generated: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, f"Automotive Diagnostic Log Analyzer · {source} · generated {generated}")
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def render_pdf(report: dict, trip_logs: pd.DataFrame, trip_scored: pd.DataFrame) -> bytes:
    """Build the PDF report and return it as bytes."""
    _register_fonts()
    s = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"Diagnosis report - vehicle {report['vehicle_id']}, trip {report['trip_id']}",
                            author="Automotive Diagnostic Log Analyzer")
    width = doc.width
    trip = report["trip"]
    story = []

    # header
    when = f" · {_esc(trip['start'][:16])}" if trip["start"] else ""
    story += [
        Paragraph("Vehicle diagnosis report", s["title"]),
        Spacer(1, 3),
        Paragraph(f"Vehicle {report['vehicle_id']} · Trip {report['trip_id']}{when} · "
                  f"{fmt_time(trip['duration_s'])} min · {trip['distance_km']} km · {trip['readings']} readings",
                  s["meta"]),
        Spacer(1, 10),
        _health_block(report, s, width),
    ]

    # fault codes
    story.append(Paragraph("Fault codes", s["h2"]))
    if report["dtcs"]:
        rows = [[Paragraph(x, s["cell_head"]) for x in ("Code", "Meaning", "Severity", "When")]]
        for d in report["dtcs"]:
            when_set = f"set at {fmt_time(d['time_s'])}" if d["time_s"] is not None else "read after drive"
            rows.append([
                Paragraph(f"<b>{_esc(d['code'])}</b>", s["cell"]),
                Paragraph(f"{_esc(d['description'])}<br/><font color='#52514e' size='8'>{_esc(d['advice'])}</font>", s["cell"]),
                Paragraph(f"{_dot(SEVERITY_COLOR.get(d['severity'], '#7a7974'))} {_esc(d['severity'])}", s["cell"]),
                Paragraph(when_set, s["cell"]),
            ])
        story.append(_table(rows, [20 * mm, width - 20 * mm - 28 * mm - 32 * mm, 28 * mm, 32 * mm]))
    else:
        story.append(Paragraph("No fault codes.", s["muted"]))

    # anomalies
    story.append(Paragraph("Anomalies and likely causes", s["h2"]))
    if not report["findings"]:
        story.append(Paragraph("No unusual behaviour detected.", s["muted"]))
    for i, f in enumerate(report["findings"], start=1):
        tag = "early warning, no fault code yet" if f["early_warning"] else "linked to " + ", ".join(f["dtc_codes"])
        block = [Paragraph(f"Anomaly {i}: {_esc(f['subsystem'])}", s["h3"]),
                 Paragraph(f"{fmt_time(f['start_s'])}–{fmt_time(f['end_s'])} · {_esc(tag)}", s["small"])]
        if f["warning_before_dtc_s"] is not None and f["warning_before_dtc_s"] > 0:
            block.append(Paragraph(f"Detected <b>{f['warning_before_dtc_s']:.0f} s before</b> the fault code.", s["base"]))
        block.append(Paragraph(f"<b>Likely cause:</b> {_esc(f['cause_label'])} "
                               f"<font color='#7a7974'>({_esc(f['confidence'])} confidence)</font>", s["base"]))
        block.append(ListFlowable([ListItem(Paragraph(_esc(e), s["small"]), leftIndent=10) for e in f["evidence"]],
                                  bulletType="bullet", bulletFontName=FONT, bulletFontSize=7, leftIndent=10))
        block.append(Spacer(1, 6))
        story.append(KeepTogether(block))

    # checks
    story.append(Paragraph("Recommended checks", s["h2"]))
    if report["recommended_checks"]:
        story.append(ListFlowable([ListItem(Paragraph(_esc(c), s["base"])) for c in report["recommended_checks"]],
                                  bulletType="1", bulletFontName=FONT, bulletFontSize=9))
    else:
        story.append(Paragraph("None needed.", s["muted"]))
    if report.get("other_possible_causes"):
        story += [Spacer(1, 4), Paragraph("Other possible causes for these codes: "
                                          + _esc(", ".join(report["other_possible_causes"])), s["muted"])]

    # chart
    png = signal_chart_png(report, trip_logs, trip_scored)
    chart = Image(io.BytesIO(png), width=width, height=width * 4.6 / 7.2)
    story.append(KeepTogether([
        Paragraph("Signals", s["h2"]),
        chart,
        Paragraph("Shaded: anomaly periods. Dotted line: fault code set. Score above 1.0 = outside normal behaviour.",
                  s["muted"]),
    ]))

    # checks run + notes
    rows = [[Paragraph(("✓ " if state == "checked" else "– ") + _esc(name), s["cell"]), Paragraph(_esc(state), s["small"])]
            for name, state in report["checks_run"].items()]
    story += [Paragraph("Checks run", s["h2"]), _table(rows, [70 * mm, width - 70 * mm], header=False)]
    for note in report["notes"]:
        story += [Spacer(1, 4), Paragraph(f"Note: {_esc(note)}", s["muted"])]

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    footer = _footer(_esc(report.get("source") or "driving log"), generated)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
