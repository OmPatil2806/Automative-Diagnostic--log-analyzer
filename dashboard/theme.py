"""Colors and styles used across the dashboard (one place, so pages stay consistent).

Matches the HTML/PDF reports: one blue for data series, status colors only for
health status, always shown together with an icon and a label.
"""

# Surfaces and text
SURFACE = "#fcfcfb"
CARD = "#ffffff"
BORDER = "#e2e1dc"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#7a7974"
GRID = "#e8e7e3"

# Data series (categorical order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE = SERIES[0]

# Health status: color + icon + label
STATUS = {
    "good": {"color": "#0ca30c", "icon": "✓", "label": "Good"},
    "needs attention": {"color": "#fab219", "icon": "⚠", "label": "Needs attention"},
    "critical": {"color": "#d03b3b", "icon": "✗", "label": "Critical"},
}

# Fault code severity
SEVERITY = {
    "low": "#fab219",
    "medium": "#ec835a",
    "high": "#d03b3b",
    "critical": "#d03b3b",
    "unknown": MUTED,
}

ANOMALY_SHADE = "#d03b3b"   # shaded anomaly periods on charts (low opacity)


def plotly_layout(**overrides) -> dict:
    """Base layout for every Plotly chart in the dashboard."""
    layout = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT_2, size=12),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    layout.update(overrides)
    return layout
