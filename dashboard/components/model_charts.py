"""Plotly charts for the Model performance page (built from the saved metrics)."""

import numpy as np
import plotly.graph_objects as go

from dashboard.theme import GRID, SERIES, TEXT, TEXT_2, plotly_layout

GREY = "#a3a29d"
METHOD_STYLE = {"forest": ("Isolation Forest only", GREY), "range": ("Range score only", SERIES[1]),
                "combined": ("Combined (used)", SERIES[0])}


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(**plotly_layout(height=height, hovermode="closest", showlegend=True,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                      margin=dict(l=10, r=30, t=40, b=40), **kw))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def grouped_bars(labels: list[str], series: dict[str, tuple[list[float], str]], x_title: str) -> go.Figure:
    """Horizontal grouped bars in %, one color per series (with legend and value labels)."""
    fig = go.Figure()
    for name, (values, color) in series.items():
        pct = [v * 100 if v is not None else None for v in values]
        fig.add_trace(go.Bar(y=labels, x=pct, name=name, orientation="h", marker=dict(color=color),
                             text=[f"{v:.0f}%" if v is not None else "" for v in pct], textposition="outside",
                             cliponaxis=False, hovertemplate=f"{name}<br>%{{y}}: %{{x:.0f}}%<extra></extra>"))
    fig = _layout(fig, max(260, 46 * len(labels) * max(1, len(series)) // 2 + 90), barmode="group",
                  xaxis_title=x_title)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 112])
    return fig


def warning_strip(warnings: dict[str, list[float]], names: dict[str, str]) -> go.Figure:
    """One dot per detected trip: seconds between the first alarm and the fault code (positive = earlier)."""
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, (fault, values) in enumerate(warnings.items()):
        vals = np.clip(values, -300, 300)
        y = i + rng.uniform(-0.18, 0.18, len(vals))
        fig.add_trace(go.Scatter(x=vals, y=y, mode="markers", showlegend=False,
                                 marker=dict(size=9, color=[SERIES[0] if v > 0 else GREY for v in vals]),
                                 customdata=values, hovertemplate=f"{names[fault]}: %{{customdata:.0f}} s<extra></extra>"))
        if len(values):
            m = float(np.median(values))
            fig.add_trace(go.Scatter(x=[m, m], y=[i - 0.32, i + 0.32], mode="lines", showlegend=False,
                                     line=dict(color=TEXT, width=3), hovertemplate=f"median {m:.0f} s<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=1, dash="dash"))
    fig.add_annotation(x=8, y=-0.75, text="before the fault code →", showarrow=False, xanchor="left",
                       font=dict(color=TEXT_2, size=11))
    fig.add_annotation(x=-8, y=-0.75, text="← after", showarrow=False, xanchor="right", font=dict(color=TEXT_2, size=11))
    fig = _layout(fig, 380, xaxis_title="seconds before the code (capped at ±300) · bar = median")
    fig.update_xaxes(range=[-330, 330])
    fig.update_yaxes(tickvals=list(range(len(warnings))), ticktext=[names[f] for f in warnings],
                     autorange="reversed", showgrid=False, range=[len(warnings) - 0.5, -1])
    return fig


def persistence_tradeoff(grid: list[dict]) -> go.Figure:
    """False alarms vs detection for each alarm rule; the rule in use is labeled."""
    x = [g["false_alarm_rate"] * 100 for g in grid]
    fig = go.Figure()
    for key, name, color in [("detected", "Faulty trips detected", SERIES[0]), ("before_dtc", "Detected before the code", SERIES[2])]:
        fig.add_trace(go.Scatter(x=x, y=[g[key] * 100 for g in grid], mode="lines+markers", name=name,
                                 line=dict(color=color, width=2), marker=dict(size=9),
                                 customdata=[f"{g['ratio']:.0%} of last {g['window']}" for g in grid],
                                 hovertemplate="%{customdata}<br>%{y:.0f}%<extra>" + name + "</extra>"))
    loosest = max(grid, key=lambda g: g["false_alarm_rate"])
    for g in grid:
        if g["used"] or g is grid[0] or g is grid[-1]:
            right_edge = g is loosest
            fig.add_annotation(x=g["false_alarm_rate"] * 100, y=g["before_dtc"] * 100,
                               xanchor="right" if right_edge else "left", yanchor="top",
                               xshift=-6 if right_edge else 8, yshift=-6,
                               text=f"{g['ratio']:.0%} of last {g['window']}" + (" (used)" if g["used"] else ""),
                               showarrow=False, font=dict(color=TEXT if g["used"] else TEXT_2, size=11))
    fig = _layout(fig, 380, xaxis_title="normal trips with a false alarm (%)", yaxis_title="% of faulty trips")
    fig.update_yaxes(range=[0, 105])
    return fig
