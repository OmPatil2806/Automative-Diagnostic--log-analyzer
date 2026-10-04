"""Diagnose a driving log and write the report.

Usage:
    python scripts/run_pipeline.py --log data/samples/vacuum_leak.csv
    python scripts/run_pipeline.py --log data/samples/vacuum_leak.csv --dtc-file data/samples/vacuum_leak_dtc.csv
    python scripts/run_pipeline.py --log my_trip.csv --dtc P0171 P0420
    python scripts/run_pipeline.py --log my_trip.csv --obd-response "43 01 71 04 20"

Input log: CSV with time_ms, speed_kmh, engine_rpm (required) and any of
maf_gs, absolute_load_pct, stft_b1_pct, ltft_b1_pct, hv_battery_voltage_v,
hv_battery_current_a, hv_battery_soc_pct (optional). Raw VED column names
also work.

Writes <name>_report.txt / .json / .html / .pdf to reports/ (or --out).
Choose formats with --format, e.g. --format pdf html.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.config import REPORTS_DIR  # noqa: E402
from analyzer.pipeline import DEFAULT_MODEL_PATH, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, type=Path, help="driving log CSV")
    parser.add_argument("--dtc", nargs="*", default=[], help="fault codes read from the car, e.g. P0171 P0420")
    parser.add_argument("--obd-response", help='raw OBD-II mode 03 hex, e.g. "43 01 71 04 20"')
    parser.add_argument("--dtc-file", type=Path, help="CSV with columns code,time_ms")
    parser.add_argument("--vehicle-id", type=int, help="VED vehicle id, to use its learned baseline")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="trained model file")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="output folder")
    parser.add_argument("--offline", action="store_true", help="embed the chart library in the HTML (no internet needed)")
    parser.add_argument("--format", nargs="+", choices=["txt", "json", "html", "pdf"],
                        default=["txt", "json", "html", "pdf"])
    args = parser.parse_args()

    try:
        results = run(args.log, dtc_codes=args.dtc, obd_response=args.obd_response,
                      dtc_file=args.dtc_file, vehicle_id=args.vehicle_id, model_path=args.model,
                      offline_html=args.offline)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")

    args.out.mkdir(parents=True, exist_ok=True)
    for result in results:
        name = args.log.stem if len(results) == 1 else f"{args.log.stem}_trip{result.diagnosis.trip_id}"
        print(result.text)
        written = []
        if "txt" in args.format:
            (args.out / f"{name}_report.txt").write_text(result.text + "\n", encoding="utf-8")
            written.append("txt")
        if "json" in args.format:
            (args.out / f"{name}_report.json").write_text(json.dumps(result.report, indent=2, default=str), encoding="utf-8")
            written.append("json")
        if "html" in args.format:
            (args.out / f"{name}_report.html").write_text(result.html, encoding="utf-8")
            written.append("html")
        if "pdf" in args.format:
            (args.out / f"{name}_report.pdf").write_bytes(result.pdf())
            written.append("pdf")
        print(f"Saved {args.out / name}_report.{{{','.join(written)}}}\n")


if __name__ == "__main__":
    main()
