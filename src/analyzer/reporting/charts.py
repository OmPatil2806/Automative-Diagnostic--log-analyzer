"""Trip chart for the HTML report: stacked panels sharing the time axis.

Panels (small multiples, one y-axis each - never a dual axis):
  1. vehicle speed (context)
  2. the signal that explains the main finding (e.g. total fuel trim)
  3. anomaly score with the normal limit (1.0)
Anomaly episodes are shaded on every panel; DTCs with a known time are marked.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIES = "#2a78d6"        # categorical slot 1 (blue)
CRITICAL = "#d03b3b"      # status: critical, used only for anomaly shading
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#fcfcfb"

# suspect subsystem -> (column source, column, panel title)
MAIN_SIGNAL = {
    "fuel": ("logs", "total_fuel_trim", "Total fuel trim (%)"),
    "air": ("scored", "maf_residual_median", "Airflow vs expected (%)"),
    "engine": ("logs", "engine_rpm", "Engine RPM"),
    "speed": ("logs", "engine_rpm", "Engine RPM"),
    "battery": ("logs", "hv_battery_voltage_v", "Battery voltage (V)"),
}


def main_signal(group: str | None, logs: pd.DataFrame, scored: pd.DataFrame) -> tuple[pd.Series, str]:
    logs = logs.assign(total_fuel_trim=logs["stft_b1_pct"] + logs["ltft_b1_pct"])
    source, column, title = MAIN_SIGNAL.get(group, ("logs", "engine_rpm", "Engine RPM"))
    frame = logs if source == "logs" else scored
    values = frame.set_index("time_ms")[column]
    if values.notna().sum() == 0:
        return logs.set_index("time_ms")["engine_rpm"], "Engine RPM"
    return values, title


def trip_figure(logs: pd.DataFrame, scored: pd.DataFrame, report: dict) -> go.Figure:
    main = report.get("main_finding")
    group = main["subsystem_key"] if main else None
    signal, signal_title = main_signal(group, logs, scored)

    panels = [
        (logs.set_index("time_ms")["speed_kmh"], "Speed (km/h)"),
        (signal, signal_title),
        (scored.set_index("time_ms")["anomaly_score"], "Anomaly score"),
    ]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[title for _, title in panels])

    t0 = logs["time_ms"].min()   # report times are relative to the trip start
    for row, (values, title) in enumerate(panels, start=1):
        minutes = (values.index - t0) / 60_000
        fig.add_trace(go.Scatter(
            x=minutes, y=values, mode="lines", name=title, line=dict(color=SERIES, width=2),
            hovertemplate=f"{title}: %{{y:.1f}}<extra></extra>", connectgaps=False,
        ), row=row, col=1)

    fig.add_hline(y=1.0, row=3, col=1, line=dict(color=TEXT_SECONDARY, width=1, dash="dash"),
                  annotation_text="normal limit", annotation_position="top left",
                  annotation_font=dict(color=TEXT_SECONDARY, size=11))

    for finding in report["findings"]:
        fig.add_vrect(x0=finding["start_s"] / 60, x1=finding["end_s"] / 60, fillcolor=CRITICAL,
                      opacity=0.12, line_width=0, row="all", col=1)

    # DTC labels sit inside the bottom panel, left of their line, clear of the panel titles
    for dtc in report["dtcs"]:
        if dtc["time_s"] is None:
            continue
        fig.add_vline(x=dtc["time_s"] / 60, line=dict(color=TEXT_PRIMARY, width=1.5, dash="dot"), row="all", col=1)
        fig.add_annotation(x=dtc["time_s"] / 60, y=0.97, yref="y3 domain", xanchor="right", yanchor="top",
                           xshift=-4, text=f"{dtc['code']} set", showarrow=False, bgcolor=SURFACE,
                           font=dict(color=TEXT_PRIMARY, size=11))

    fig.update_layout(
        height=620, showlegend=False, hovermode="x unified",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    fig.update_annotations(font_color=TEXT_SECONDARY)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_xaxes(title_text="Time into trip (min)", row=3, col=1)
    return fig
