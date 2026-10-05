"""Model performance: how well the anomaly detector finds faults, how early, and at what cost."""

import streamlit as st

from dashboard.components import model_charts as charts
from dashboard.components.kpi import kpi_row
from dashboard.components.layout import page_header
from dashboard.components.report_view import html_table, section
from dashboard.services.metrics_service import FAULT_NAMES, GROUP_NAMES, load_metrics
from dashboard.services.model_service import model_info

page_header(
    "Model performance",
    "How well the anomaly detector finds faults, how early it warns, and how often it raises a false alarm.",
)

m = load_metrics()
if m is None:
    st.warning("No evaluation results yet. Run `python scripts/export_metrics.py`.")
    st.stop()

d, combined, diag = m["dataset"], m["methods"]["combined"], m["diagnosis"]
faults = [f["fault"] for f in m["per_fault"]]
names = [FAULT_NAMES[f] for f in faults]
detected_avg = sum(f["detected"] for f in m["per_fault"]) / len(m["per_fault"])

st.caption(f"Evaluated on {d['trips']} trips the model never saw ({d['normal_trips']} normal, {d['faulty_trips']} with a "
           f"simulated fault, {d['readings']:,} readings). The faults are injected into real VED driving, so these "
           "numbers show the method works on realistic signal changes, not how it performs on real failures.")
kpi_row([
    ("Faulty trips detected", f"{detected_avg:.0%}", "Average over the six fault types"),
    ("Normal trips with a false alarm", f"{combined['false_alarm_rate']:.1%}", "Normal trips that raised an alarm"),
    ("Right root cause", f"{diag['root_cause_accuracy']:.0%}", "Faulty trips with an anomaly whose likely cause was correct"),
])
kpi_row([
    ("F1 score", f"{combined['f1']:.2f}", "Readings in fully developed faults vs normal driving"),
    ("Precision", f"{combined['precision']:.0%}", "Of the alarm readings, how many were really faulty"),
    ("Recall", f"{combined['recall']:.0%}", "Of the fully faulty readings, how many raised an alarm"),
])

tab_faults, tab_methods, tab_rule, tab_diag, tab_model = st.tabs(
    ["Per fault", "Forest vs range score", "Alarm rule", "Root cause and health score", "Model and limitations"])

with tab_faults:
    left, right = st.columns([1, 1], gap="large")
    with left:
        section("Detected, and detected before the fault code")
        per = m["per_fault"]
        st.plotly_chart(charts.grouped_bars(names, {
            "Detected": ([f["detected"] for f in per], charts.SERIES[0]),
            "Before the fault code": ([f["before_dtc"] for f in per], charts.SERIES[2]),
        }, "% of faulty trips"), width="stretch", key="perf_faults")
    with right:
        section("How early is the warning?")
        st.plotly_chart(charts.warning_strip(m["warning_times"], FAULT_NAMES), width="stretch", key="perf_warning")
    html_table([{
        "fault": FAULT_NAMES[f["fault"]], "detected": f"{f['detected']:.0%}", "before": f"{f['before_dtc']:.0%}",
        "warning": f"{f['median_warning_s']:+.0f} s" if f["median_warning_s"] is not None else "-",
        "group": GROUP_NAMES.get(f["suspect_group"], f["suspect_group"]),
    } for f in per], {"fault": "Fault", "detected": "Detected", "before": "Before the code",
                      "warning": "Median warning", "group": "Subsystem it points to"})
    st.caption("Battery, MAF, speed and vacuum-leak faults are caught before the code in roughly half the trips. "
               "Misfire and rich-injector faults are mostly caught just after the code: they must grow large "
               "before they stand out from normal driving. The window is short (median about 1.5 minutes from "
               "fault start to code).")

with tab_methods:
    section("Isolation Forest alone, range score alone, and both combined")
    rows = [{"method": charts.METHOD_STYLE[k][0], "f1": f"{v['f1']:.2f}", "precision": f"{v['precision']:.0%}",
             "recall": f"{v['recall']:.0%}", "detected": f"{sum(v['detection_rate'].values()) / len(faults):.0%}",
             "fa": f"{v['false_alarm_rate']:.1%}"} for k, v in m["methods"].items()]
    html_table(rows, {"method": "Scoring", "f1": "F1", "precision": "Precision", "recall": "Recall",
                      "detected": "Faulty trips detected", "fa": "False alarms"})
    st.plotly_chart(charts.grouped_bars(names, {
        charts.METHOD_STYLE[k][0]: ([v["detection_rate"][f] for f in faults], charts.METHOD_STYLE[k][1])
        for k, v in m["methods"].items()
    }, "faulty trips detected (%)"), width="stretch", key="perf_methods")
    st.markdown("An Isolation Forest cannot score values beyond its training range: a +20% fuel trim looks no worse "
                "than the most extreme normal value, so on its own it misses most vacuum-leak and MAF faults. The "
                "range score measures how far each feature is past its normal band and does most of the work; "
                "combining both adds a small further gain by also catching unusual combinations of features.")

with tab_rule:
    section("Alarm rule: false alarms vs early detection")
    st.plotly_chart(charts.persistence_tradeoff(m["persistence"]), width="stretch", key="perf_rule")
    rule = m["model"]["alarm_rule"]
    st.markdown(f"An alarm needs the anomaly to persist: **{rule['ratio']:.0%} of the last {rule['window']} readings**. "
                "A looser rule warns earlier but alarms on more healthy trips; a stricter one misses the early part "
                "of the window. This setting was chosen on the same synthetic set, so the results are somewhat "
                "optimistic.")

with tab_diag:
    section("Root cause, from the signals only")
    html_table([{"fault": FAULT_NAMES[f["fault"]],
                 "acc": f"{f['root_cause_accuracy']:.0%}" if f["root_cause_accuracy"] is not None else "-"}
                for f in m["per_fault"]], {"fault": "Fault", "acc": "Correct likely cause"})
    st.caption(f"{diag['root_cause_accuracy']:.0%} correct over the {diag['faulty_with_anomaly']} faulty trips where an "
               "anomaly was found. The cause is read from the signals (for example: fuel trims high and airflow normal "
               "means a vacuum leak), not from the fault code, so it also works before any code is set.")
    section("Health score")
    kpi_row([("Normal trips rated Good", f"{diag['normal_good_rate']:.0%}", None),
             ("Normal trips, mean score", f"{diag['normal_mean_score']:.0f}", None),
             ("Faulty trips flagged", f"{diag['faulty_flagged_rate']:.0%}", "Rated Needs attention or Critical"),
             ("Faulty trips, mean score", f"{diag['faulty_mean_score']:.0f}", None)])
    st.caption("Every simulated faulty trip sets a fault code, so flagging them is easy here; the root-cause accuracy "
               "is the more meaningful number.")

with tab_model:
    info = model_info()
    section("The model")
    html_table([
        {"k": "Approach", "v": "Unsupervised anomaly detection, trained on normal driving only"},
        {"k": "Detectors", "v": ", ".join(GROUP_NAMES.get(g, g) for g in info["detectors"]) + " (one per subsystem)"},
        {"k": "Scoring", "v": "Higher of an Isolation Forest score and a range score; 1.0 = edge of normal"},
        {"k": "Per-vehicle baselines", "v": f"{info['vehicles_with_baseline']} vehicles (airflow, fuel trim, battery)"},
        {"k": "Alarm rule", "v": info["alarm_rule"]},
        {"k": "Training data", "v": f"{d['training_trips']} normal VED trips (week 1), none of them in the evaluation"
                                    if d.get("training_trips") else "Normal VED trips (week 1)"},
        {"k": "Model file", "v": f"{info['file']} ({info['size_mb']} MB, {info['modified']})"},
    ], {"k": "", "v": ""})
    section("Limitations")
    st.markdown(
        "- **Simulated faults.** Real failures may look different.\n"
        "- **Tuned on the test set.** The alarm rule was chosen on the same synthetic trips.\n"
        "- **One week of training data**, in which only about 15% of trips record fuel trims.\n"
        "- **About one reading per second** limits misfire detection.\n"
        "- **Most false alarms come from the battery detector**, whose linear voltage model ignores temperature."
    )
    st.caption(f"Results generated {m['generated']} by `python scripts/export_metrics.py`.")
