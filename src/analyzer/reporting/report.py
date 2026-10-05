"""Build a readable diagnosis report (summary, faults, anomalies, health score).

`build_report` produces a JSON-ready dict; `render_text` and `render_html`
turn it into a terminal report and a standalone HTML page with a chart.
"""

import html
from datetime import datetime

import pandas as pd

from analyzer.analysis.diagnosis import TripDiagnosis
from analyzer.analysis.health_score import ATTENTION, CRITICAL, GOOD
from analyzer.features.engineering import FEATURE_GROUPS

SUBSYSTEMS = {
    "engine": "Engine (RPM stability)",
    "speed": "Vehicle speed sensor",
    "fuel": "Fuel system (fuel trims)",
    "air": "Air intake (MAF airflow)",
    "battery": "Hybrid / EV battery",
}
MIN_CHECK_COVERAGE = 0.2   # share of rows a subsystem needs to count as checked
MAX_CHECKS = 6

STATUS_ICON = {GOOD: "✓", ATTENTION: "⚠", CRITICAL: "✗"}
STATUS_TEXT = {GOOD: "GOOD", ATTENTION: "NEEDS ATTENTION", CRITICAL: "CRITICAL"}


def fmt_time(seconds: float | None) -> str:
    """Seconds into the trip -> 'm:ss'."""
    if seconds is None or pd.isna(seconds):
        return "-"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# raw signals each detector needs (to tell "signal missing" from "log too short")
GROUP_SIGNALS = {
    "engine": ["engine_rpm", "speed_kmh"],
    "speed": ["speed_kmh", "engine_rpm"],
    "fuel": ["stft_b1_pct", "ltft_b1_pct"],
    "air": ["maf_gs", "absolute_load_pct", "engine_rpm"],
    "battery": ["hv_battery_voltage_v", "hv_battery_current_a", "hv_battery_soc_pct"],
}


def _checks_run(scored: pd.DataFrame, detector, trip_logs: pd.DataFrame | None = None) -> dict[str, str]:
    checks = {}
    for group in FEATURE_GROUPS:
        column = f"score_{group}"
        if group not in detector.models:
            checks[group] = "not available (no trained model)"
        elif column not in scored or scored[column].notna().mean() < MIN_CHECK_COVERAGE:
            has_signals = trip_logs is not None and all(
                c in trip_logs and trip_logs[c].notna().any() for c in GROUP_SIGNALS.get(group, []))
            checks[group] = ("not checked (too few readings; needs about a minute of driving)" if has_signals
                             else "not checked (signals missing in this log)")
        else:
            checks[group] = "checked"
    return checks


def _summary(diagnosis: TripDiagnosis, findings: list[dict], main: dict | None) -> str:
    if not diagnosis.dtcs and not findings:
        return "No fault codes and no unusual behaviour found."
    if not diagnosis.dtcs:
        return (f"No fault codes, but {len(findings)} early warning(s). "
                f"Most likely: {main['cause_label']}. Keep an eye on it.")
    codes = ", ".join(d["code"] for d in diagnosis.dtcs)
    if main is None:
        return f"Fault code(s) {codes} set. No matching unusual behaviour in this trip's data."
    text = f"Fault code(s) {codes} set. Likely cause: {main['cause_label']}."
    if main["warning_before_dtc_s"] and main["warning_before_dtc_s"] > 0:
        text += f" The problem showed in the signals {main['warning_before_dtc_s']:.0f} s before the code was set."
    return text


def build_report(
    diagnosis: TripDiagnosis,
    trip_logs: pd.DataFrame,
    trip_scored: pd.DataFrame,
    detector,
    source: str = "",
) -> dict:
    """Assemble everything known about one trip into a JSON-ready dict."""
    t0 = int(trip_logs["time_ms"].min())
    dt_h = trip_logs["time_ms"].diff().fillna(0).clip(upper=10_000) / 3_600_000
    start = trip_logs["timestamp"].min() if "timestamp" in trip_logs else None
    vid = diagnosis.vehicle_id
    baseline = detector.baseline
    known_vehicle = vid in baseline.maf_k or vid in baseline.battery_coef or vid in baseline.stft_offset

    findings = []
    for f in diagnosis.findings:
        findings.append({
            "start_s": (f.start_ms - t0) / 1000,
            "end_s": (f.end_ms - t0) / 1000,
            "duration_s": f.duration_s,
            "subsystem_key": f.suspect_group,
            "subsystem": SUBSYSTEMS.get(f.suspect_group, f.suspect_group),
            "peak_score": round(f.peak_score, 2),
            "early_warning": f.is_early_warning,
            "dtc_codes": f.dtc_codes,
            "warning_before_dtc_s": f.warning_before_dtc_s,
            "cause": f.root_cause.cause,
            "cause_label": f.root_cause.label,
            "confidence": f.root_cause.confidence,
            "evidence": f.root_cause.evidence,
            "checks": f.root_cause.checks,
        })
    main_obj = diagnosis.main_finding
    main = findings[diagnosis.findings.index(main_obj)] if main_obj else None

    dtcs = [{
        **d,
        "time_s": None if d["time_ms"] is None else (d["time_ms"] - t0) / 1000,
    } for d in diagnosis.dtcs]

    # checks come from the diagnosed causes; if nothing was diagnosed, fall back to the DTCs' causes
    checks = [c for f in ([main] if main else []) + [f for f in findings if f is not main] for c in f["checks"]]
    dtc_causes = list(dict.fromkeys(c for d in dtcs for c in d["possible_causes"]))
    if not checks:
        checks = [f"Check: {c}" for c in dtc_causes]
        dtc_causes = []
    recommended = list(dict.fromkeys(checks))[:MAX_CHECKS]

    notes = []
    if not known_vehicle:
        notes.append("Vehicle not seen in training: airflow is compared with earlier in this trip, "
                     "and battery checks are limited.")
    if any(d["time_s"] is None for d in dtcs):
        notes.append("Fault code times are unknown (read after the drive), so early warning time is not shown.")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "vehicle_id": vid,
        "trip_id": diagnosis.trip_id,
        "trip": {
            "start": None if start is None or pd.isna(start) else str(start),
            "duration_s": (trip_logs["time_ms"].max() - t0) / 1000,
            "distance_km": round(float((trip_logs["speed_kmh"].fillna(0) * dt_h).sum()), 1),
            "readings": len(trip_logs),
        },
        "health": {
            "score": diagnosis.health_score,
            "status": diagnosis.status,
            "summary": _summary(diagnosis, findings, main),
        },
        "dtcs": dtcs,
        "findings": findings,
        "main_finding": main,
        "recommended_checks": recommended,
        "other_possible_causes": dtc_causes,
        "checks_run": {SUBSYSTEMS[g]: v for g, v in _checks_run(trip_scored, detector, trip_logs).items()},
        "notes": notes,
    }


# --- text -----------------------------------------------------------------

def render_text(report: dict, width: int = 64) -> str:
    line, thin = "═" * width, "─" * width
    h, trip = report["health"], report["trip"]
    out = [line, " VEHICLE DIAGNOSIS REPORT",
           f" Vehicle {report['vehicle_id']}   Trip {report['trip_id']}"
           + (f"   {trip['start'][:16]}" if trip["start"] else ""),
           f" {fmt_time(trip['duration_s'])} min drive, {trip['distance_km']} km, {trip['readings']} readings",
           line,
           f" HEALTH SCORE:  {h['score']} / 100   {STATUS_ICON[h['status']]} {STATUS_TEXT[h['status']]}",
           "", *_wrap(h["summary"], width, " "), ""]

    out.append(" FAULT CODES")
    if not report["dtcs"]:
        out.append("  none")
    for d in report["dtcs"]:
        when = f"set at {fmt_time(d['time_s'])}" if d["time_s"] is not None else "read after drive"
        out.append(f"  {d['code']}  {d['description']}")
        out.append(f"         {d['severity'].upper()} · {when}")
        out += _wrap(f"→ {d['advice']}", width, "         ")
    out.append("")

    out.append(" ANOMALIES DETECTED")
    if not report["findings"]:
        out.append("  none")
    for i, f in enumerate(report["findings"], start=1):
        tag = "EARLY WARNING (no fault code yet)" if f["early_warning"] else f"linked to {', '.join(f['dtc_codes'])}"
        out.append(f"  {i}. {f['subsystem']}   {fmt_time(f['start_s'])}–{fmt_time(f['end_s'])}   {tag}")
        if f["warning_before_dtc_s"] is not None and f["warning_before_dtc_s"] > 0:
            out.append(f"     detected {f['warning_before_dtc_s']:.0f} s BEFORE the fault code")
        out.append(f"     Likely cause: {f['cause_label']} ({f['confidence']} confidence)")
        for e in f["evidence"]:
            out += _wrap(f"- {e}", width, "       ")
    out.append("")

    out.append(" RECOMMENDED CHECKS")
    if not report["recommended_checks"]:
        out.append("  none needed")
    for i, c in enumerate(report["recommended_checks"], start=1):
        out.append(f"  {i}. {c}")
    if report["other_possible_causes"]:
        out += ["", " OTHER POSSIBLE CAUSES FOR THESE CODES"]
        out += _wrap(", ".join(report["other_possible_causes"]), width, "  ")
    out += ["", thin, " Checks run:"]
    for name, state in report["checks_run"].items():
        out.append(f"  {'✓' if state == 'checked' else '–'} {name}: {state}")
    for n in report["notes"]:
        out += _wrap(f"Note: {n}", width, " ")
    out.append(line)
    return "\n".join(out)


def _wrap(text: str, width: int, indent: str) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent + "  ")


# --- HTML -----------------------------------------------------------------

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --card:#ffffff; --border:#e2e1dc;
  --text:#0b0b0b; --text-2:#52514e; --muted:#7a7974;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 10px; }
.meta { color: var(--text-2); margin: 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.health { display: flex; gap: 20px; align-items: center; margin-top: 18px; flex-wrap: wrap; }
.score { font-size: 44px; font-weight: 700; line-height: 1; }
.score small { font-size: 16px; color: var(--text-2); font-weight: 400; }
.badge { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; padding: 4px 12px 4px 8px;
  border-radius: 999px; border: 1px solid var(--border); }
.dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex: none; }
.summary { flex: 1 1 320px; margin: 0; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--text-2); font-weight: 600; font-size: 13px; }
code { font: 600 14px ui-monospace, Menlo, Consolas, monospace; }
.finding { margin-bottom: 12px; } .finding h3 { font-size: 15px; margin: 0 0 6px; }
.finding ul { margin: 6px 0 0; padding-left: 20px; color: var(--text-2); }
.tag { font-size: 12px; font-weight: 600; color: var(--text-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 1px 6px; margin-left: 6px; white-space: nowrap; }
ol { margin: 0; padding-left: 22px; } .muted { color: var(--muted); font-size: 13px; }
.chart { padding: 8px; overflow-x: auto; }
@media (max-width: 560px) { th:nth-child(4), td:nth-child(4) { display: none; } .score { font-size: 36px; } }
"""

STATUS_COLOR = {GOOD: "var(--good)", ATTENTION: "var(--warning)", CRITICAL: "var(--critical)"}
SEVERITY_COLOR = {"low": "var(--warning)", "medium": "var(--serious)", "high": "var(--critical)",
                  "critical": "var(--critical)", "unknown": "var(--muted)"}


def render_html(report: dict, trip_logs: pd.DataFrame, trip_scored: pd.DataFrame, offline: bool = False) -> str:
    """Standalone HTML page. `offline=True` embeds plotly.js (~4.5 MB) instead of loading it from a CDN."""
    from analyzer.reporting.charts import trip_figure

    e = html.escape
    h, trip = report["health"], report["trip"]
    chart = trip_figure(trip_logs, trip_scored, report).to_html(
        full_html=False, include_plotlyjs=True if offline else "cdn", config={"displaylogo": False, "responsive": True})

    dtc_rows = "".join(
        f"<tr><td><code>{e(d['code'])}</code></td><td>{e(d['description'])}</td>"
        f"<td><span class='dot' style='background:{SEVERITY_COLOR.get(d['severity'])}'></span> {e(d['severity'])}</td>"
        f"<td>{'set at ' + fmt_time(d['time_s']) if d['time_s'] is not None else 'read after drive'}</td></tr>"
        for d in report["dtcs"]
    ) or "<tr><td colspan='4' class='muted'>No fault codes</td></tr>"

    findings = ""
    for i, f in enumerate(report["findings"], start=1):
        tag = "early warning, no fault code yet" if f["early_warning"] else "linked to " + ", ".join(f["dtc_codes"])
        lead = (f"<p>Detected <strong>{f['warning_before_dtc_s']:.0f} s before</strong> the fault code.</p>"
                if f["warning_before_dtc_s"] is not None and f["warning_before_dtc_s"] > 0 else "")
        evidence = "".join(f"<li>{e(x)}</li>" for x in f["evidence"])
        findings += (
            f"<div class='card finding'><h3>Anomaly {i}: {e(f['subsystem'])}"
            f"<span class='tag'>{fmt_time(f['start_s'])}–{fmt_time(f['end_s'])}</span>"
            f"<span class='tag'>{e(tag)}</span></h3>{lead}"
            f"<p><strong>Likely cause:</strong> {e(f['cause_label'])} "
            f"<span class='muted'>({e(f['confidence'])} confidence)</span></p><ul>{evidence}</ul></div>"
        )
    findings = findings or "<p class='muted'>No unusual behaviour detected.</p>"

    checks = "".join(f"<li>{e(c)}</li>" for c in report["recommended_checks"]) or "<li>None needed</li>"
    other = report["other_possible_causes"]
    other = (f"<p class='muted'>Other possible causes for these codes: {e(', '.join(other))}</p>" if other else "")
    checks_run = "".join(
        f"<tr><td>{'✓' if s == 'checked' else '–'} {e(n)}</td><td class='muted'>{e(s)}</td></tr>"
        for n, s in report["checks_run"].items()
    )
    notes = "".join(f"<p class='muted'>Note: {e(n)}</p>" for n in report["notes"])
    when = f" · {e(trip['start'][:16])}" if trip["start"] else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnosis Report</title><style>{CSS}</style></head>
<body><main>
<h1>Vehicle diagnosis report</h1>
<p class="meta">Vehicle {report['vehicle_id']} · Trip {report['trip_id']}{when} ·
{fmt_time(trip['duration_s'])} min · {trip['distance_km']} km · {trip['readings']} readings</p>

<div class="card health">
  <div class="score">{h['score']}<small> / 100</small></div>
  <span class="badge"><span class="dot" style="background:{STATUS_COLOR[h['status']]}"></span>
    {STATUS_ICON[h['status']]} {STATUS_TEXT[h['status']]}</span>
  <p class="summary">{e(h['summary'])}</p>
</div>

<h2>Fault codes</h2>
<div class="card"><table><thead><tr><th>Code</th><th>Meaning</th><th>Severity</th><th>When</th></tr></thead>
<tbody>{dtc_rows}</tbody></table></div>

<h2>Anomalies and likely causes</h2>
{findings}

<h2>Recommended checks</h2>
<div class="card"><ol>{checks}</ol>{other}</div>

<h2>Signals</h2>
<div class="card chart">{chart}
<p class="muted">Shaded: anomaly periods. Dotted line: fault code set. Score above 1.0 = outside normal behaviour.</p></div>

<h2>Checks run</h2>
<div class="card"><table><tbody>{checks_run}</tbody></table>{notes}</div>
<p class="muted">Generated {e(report['generated_at'])} from {e(report['source'])}.</p>
</main></body></html>
"""
