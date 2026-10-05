"""Plotly charts for the Data exploration page (built from the saved EDA summary)."""

import numpy as np
import plotly.graph_objects as go

from dashboard.theme import BLUE, GRID, SERIES, TEXT, plotly_layout

BAND = "rgba(42,120,214,0.12)"   # normal range (1st to 99th percentile)


def _base(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(**plotly_layout(height=height, hovermode="closest", margin=dict(l=10, r=30, t=10, b=40),
                                      **layout))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def hbar(labels: list[str], values: list[float], fmt: str = "{:.0f}", height: int | None = None,
         x_title: str = "") -> go.Figure:
    """Horizontal bars, largest first, with the value at the end of each bar."""
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=BLUE),
                           text=[fmt.format(v) for v in values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig = _base(fig, height or max(180, 30 * len(labels) + 50), xaxis_title=x_title)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, max(values) * 1.15 if values else 1])
    return fig


def histogram(hist: dict, x_title: str, show_band: bool = True, height: int = 280) -> go.Figure:
    """Bars from precomputed edges/counts, with the normal range shaded."""
    edges = np.array(hist["edges"])
    centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    fig = go.Figure(go.Bar(x=centers, y=hist["counts"], width=widths, marker=dict(color=BLUE, line=dict(width=0)),
                           hovertemplate="%{x:.1f}: %{y:,}<extra></extra>"))
    if show_band:
        fig.add_vrect(x0=hist["p1"], x1=hist["p99"], fillcolor=BAND, line_width=0, layer="below")
        fig.add_vline(x=hist["median"], line=dict(color=TEXT, width=1.2, dash="dot"))
    fig = _base(fig, height, xaxis_title=x_title, bargap=0)
    fig.update_yaxes(title_text="count")
    return fig


def weekly_coverage(weekly: list[dict]) -> go.Figure:
    """Share of trips recording fuel trims and airflow, per week (two series, legend + end labels)."""
    weeks = [w["week"] for w in weekly]
    fig = go.Figure()
    for key, name, color in [("airflow", "Airflow (MAF)", SERIES[0]), ("fuel_trims", "Fuel trims", SERIES[1])]:
        values = [w[key] for w in weekly]
        fig.add_trace(go.Scatter(x=weeks, y=values, name=name, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=9),
                                 hovertemplate=f"{name}, %{{x}}: %{{y:.0f}}%<extra></extra>"))
    fig = _base(fig, 300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                yaxis_title="% of trips")
    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(tickangle=0)
    return fig


def vehicle_dots(values: list[float], x_title: str) -> go.Figure:
    """One dot per vehicle, sorted (a strip of usual values)."""
    fig = go.Figure(go.Scatter(x=values, y=list(range(len(values))), mode="markers",
                               marker=dict(color=BLUE, size=9), hovertemplate="%{x:.0f}<extra></extra>"))
    fig = _base(fig, 280, xaxis_title=x_title)
    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="vehicles")
    return fig


def airflow_scatter(airflow: list[dict]) -> go.Figure:
    """MAF vs RPM x load for a few vehicles, each with its fitted line (slope = engine constant)."""
    fig = go.Figure()
    for i, v in enumerate(airflow):
        color = SERIES[i]
        fig.add_trace(go.Scattergl(x=v["rpm_x_load"], y=v["maf_gs"], mode="markers", showlegend=False,
                                   marker=dict(color=color, size=4, opacity=0.35),
                                   hovertemplate=f"vehicle {v['vehicle_id']}<br>%{{x:.0f}} → %{{y:.1f}} g/s<extra></extra>"))
        top = float(np.quantile(v["rpm_x_load"], 0.99)) if v["rpm_x_load"] else 0
        fig.add_trace(go.Scatter(x=[0, top], y=[0, v["slope"] * top], mode="lines", line=dict(color=color, width=2.5),
                                 name=f"vehicle {v['vehicle_id']}: maf ≈ {v['slope']:.4f} × rpm × load"))
    fig = _base(fig, 440, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title="RPM × load", yaxis_title="airflow, MAF (g/s)")
    fig.update_layout(margin=dict(l=10, r=30, t=70, b=40))
    return fig


def battery_scatter(battery: dict) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=battery["hv_battery_current_a"], y=battery["hv_battery_voltage_v"], mode="markers",
                                 marker=dict(color=BLUE, size=4, opacity=0.35),
                                 hovertemplate="%{x:.0f} A → %{y:.0f} V<extra></extra>"))
    return _base(fig, 400, xaxis_title="battery current (A, negative = discharging)",
                 yaxis_title="battery voltage (V)")
