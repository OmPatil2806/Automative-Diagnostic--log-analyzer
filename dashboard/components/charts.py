"""Interactive Plotly charts for the dashboard.

Small multiples (one panel and one y-axis per signal, shared time axis), never
a dual axis. Each detector keeps the same color everywhere, anomaly periods are
shaded on every panel, and fault codes with a known time are marked.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer.reporting.charts import trip_figure
from dashboard.theme import ANOMALY_SHADE, BLUE, GRID, SERIES, TEXT, TEXT_2, plotly_layout

# column -> label shown in pickers and panel titles
SIGNALS = {
    "speed_kmh": "Speed (km/h)",
    "engine_rpm": "Engine RPM",
    "total_fuel_trim": "Total fuel trim (%)",
    "stft_b1_pct": "Short-term fuel trim (%)",
    "ltft_b1_pct": "Long-term fuel trim (%)",
    "maf_gs": "Airflow, MAF (g/s)",
    "absolute_load_pct": "Engine load (%)",
    "hv_battery_voltage_v": "Battery voltage (V)",
    "hv_battery_current_a": "Battery current (A)",
    "hv_battery_soc_pct": "Battery charge (%)",
}

# detector -> fixed color, so a detector never changes color between charts
DETECTOR_COLORS = {"engine": SERIES[0], "speed": SERIES[1], "fuel": SERIES[2], "air": SERIES[3], "battery": SERIES[4]}
DETECTOR_NAMES = {"engine": "Engine", "speed": "Speed sensor", "fuel": "Fuel system",
                  "air": "Air intake", "battery": "Battery"}

# signals that best show each detector's fault
KEY_SIGNAL = {"fuel": "total_fuel_trim", "air": "maf_gs", "engine": "engine_rpm",
              "speed": "speed_kmh", "battery": "hv_battery_voltage_v"}


def with_derived(logs: pd.DataFrame) -> pd.DataFrame:
    """Add total fuel trim (short + long term)."""
    return logs.assign(total_fuel_trim=logs["stft_b1_pct"] + logs["ltft_b1_pct"])


def available_signals(logs: pd.DataFrame) -> list[str]:
    """Signals this log actually contains."""
    logs = with_derived(logs)
    return [c for c in SIGNALS if c in logs and logs[c].notna().any()]


def default_signals(logs: pd.DataFrame, report: dict) -> list[str]:
    """Speed, RPM and the signal behind the main finding."""
    chosen = ["speed_kmh", "engine_rpm"]
    main = report.get("main_finding")
    if main and KEY_SIGNAL.get(main["subsystem_key"]) not in chosen:
        chosen.append(KEY_SIGNAL[main["subsystem_key"]])
    return [c for c in chosen if c in available_signals(logs)]


def _mark_events(fig: go.Figure, report: dict, rows: int, window: tuple[float, float] | None) -> None:
    lo, hi = window or (float("-inf"), float("inf"))
    for f in report["findings"]:
        x0, x1 = max(f["start_s"] / 60, lo), min(f["end_s"] / 60, hi)
        if x0 < x1:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=ANOMALY_SHADE, opacity=0.1, line_width=0, row="all", col=1)
    for d in report["dtcs"]:
        if d["time_s"] is not None and lo <= d["time_s"] / 60 <= hi:
            fig.add_vline(x=d["time_s"] / 60, line=dict(color=TEXT, width=1.5, dash="dot"), row="all", col=1)
            fig.add_annotation(x=d["time_s"] / 60, y=1, yref=f"y{rows if rows > 1 else ''} domain", xanchor="right",
                               yanchor="top", xshift=-4, text=f"{d['code']} set", showarrow=False,
                               font=dict(color=TEXT, size=11), bgcolor="rgba(252,252,251,0.8)")


def _minutes(frame: pd.DataFrame, t0: int) -> pd.Series:
    return (frame["time_ms"] - t0) / 60_000


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_annotations(font_color=TEXT_2)


def trip_overview(logs: pd.DataFrame, scored: pd.DataFrame, report: dict) -> go.Figure:
    """Speed, key signal and anomaly score: the same chart as the HTML report."""
    fig = trip_figure(logs, scored, report)
    fig.update_layout(height=560, margin=dict(l=50, r=20, t=40, b=40))
    return fig


def signals_figure(logs: pd.DataFrame, report: dict, signals: list[str],
                   window: tuple[float, float] | None = None) -> go.Figure:
    """One panel per chosen signal, sharing the time axis (minutes into the trip)."""
    logs = with_derived(logs)
    t0 = logs["time_ms"].min()
    minutes = _minutes(logs, t0)
    if window:
        logs, minutes = logs[minutes.between(*window)], minutes[minutes.between(*window)]
    rows = max(len(signals), 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06 if rows > 1 else 0.1,
                        subplot_titles=[SIGNALS[s] for s in signals] or [""])
    for row, column in enumerate(signals, start=1):
        fig.add_trace(go.Scatter(x=minutes, y=logs[column], mode="lines", name=SIGNALS[column],
                                 line=dict(color=BLUE, width=1.8),
                                 hovertemplate=f"{SIGNALS[column]}: %{{y:.1f}}<extra></extra>"), row=row, col=1)
    _mark_events(fig, report, rows, window)
    fig.update_layout(**plotly_layout(height=max(220, 170 * rows + 60)))
    fig.update_xaxes(title_text="Time into trip (min)", row=rows, col=1)
    _style_axes(fig)
    return fig


def detector_scores_figure(logs: pd.DataFrame, scored: pd.DataFrame, report: dict,
                           window: tuple[float, float] | None = None) -> go.Figure:
    """Each detector's score over time (one shared scale: 1.0 = edge of normal behaviour)."""
    t0 = logs["time_ms"].min()
    minutes = _minutes(scored, t0)
    if window:
        scored, minutes = scored[minutes.between(*window)], minutes[minutes.between(*window)]
    fig = go.Figure()
    for group, color in DETECTOR_COLORS.items():
        column = f"score_{group}"
        if column in scored and scored[column].notna().any():
            fig.add_trace(go.Scatter(x=minutes, y=scored[column].clip(upper=6), mode="lines",
                                     name=DETECTOR_NAMES[group], line=dict(color=color, width=1.8),
                                     hovertemplate=f"{DETECTOR_NAMES[group]}: %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=TEXT_2, width=1, dash="dash"),
                  annotation_text="normal limit", annotation_position="top left",
                  annotation_font=dict(color=TEXT_2, size=11))
    _mark_events(fig, report, 1, window)
    fig.update_layout(**plotly_layout(
        height=340, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="score (capped at 6)", xaxis_title="Time into trip (min)",
        margin=dict(l=50, r=20, t=30, b=40),
    ))
    _style_axes(fig)
    return fig


# --- overview charts (many trips) -----------------------------------------

def _status_meta():
    from dashboard.theme import STATUS
    return STATUS


def status_figure(summary: pd.DataFrame) -> go.Figure:
    """How many trips are in each health status (status color + text label on every bar)."""
    status = _status_meta()
    counts = summary["status"].value_counts()
    labels = [status[s]["label"] for s in status]
    values = [int(counts.get(s, 0)) for s in status]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=[status[s]["color"] for s in status]),
        text=[f"{v}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} trip(s)<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(height=220, hovermode="closest", margin=dict(l=10, r=30, t=10, b=30)))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, dtick=1, rangemode="tozero")
    return fig


def health_scores_figure(summary: pd.DataFrame) -> go.Figure:
    """Health score per trip, colored by status, with the 50 and 80 status thresholds."""
    status = _status_meta()
    ordered = summary.sort_values("health_score", ascending=True)
    fig = go.Figure()
    for key, meta in status.items():
        rows = ordered[ordered["status"] == key]
        if len(rows):
            fig.add_trace(go.Bar(
                x=rows["health_score"], y=rows["trip"], orientation="h", name=meta["label"],
                marker=dict(color=meta["color"]), text=rows["health_score"], textposition="inside",
                insidetextanchor="end", textfont=dict(color=TEXT), cliponaxis=False,
                hovertemplate="%{y}: %{x}/100<extra>" + meta["label"] + "</extra>",
            ))
    for threshold in (50, 80):
        fig.add_vline(x=threshold, line=dict(color=TEXT_2, width=1, dash="dash"), layer="below")
    fig.update_layout(**plotly_layout(
        height=max(220, 34 * len(ordered) + 80), hovermode="closest", showlegend=True, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=30, t=30, b=30),
    ))
    fig.update_yaxes(categoryorder="array", categoryarray=list(ordered["trip"]), showgrid=False, automargin=True)
    fig.update_xaxes(range=[0, 100], gridcolor=GRID, zeroline=False, title_text="health score")
    return fig


def causes_figure(summary: pd.DataFrame) -> go.Figure:
    """How often each likely cause was found."""
    counts = summary.loc[summary["likely_cause"] != "", "likely_cause"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=[c.split(" (")[0] for c in counts.index], orientation="h",
        marker=dict(color=BLUE), text=counts.values, textposition="outside", cliponaxis=False,
        customdata=counts.index, hovertemplate="%{customdata}: %{x} trip(s)<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(height=max(180, 32 * len(counts) + 60), hovermode="closest",
                                      margin=dict(l=10, r=30, t=10, b=30)))
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, dtick=1, rangemode="tozero")
    return fig
