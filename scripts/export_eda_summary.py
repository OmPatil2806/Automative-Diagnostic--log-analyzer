"""Summarize the VED data for the dashboard's Data exploration page.

Usage:
    python scripts/export_eda_summary.py            # first 4 weeks (default)
    python scripts/export_eda_summary.py --weeks 8

Reads the downloaded weekly VED files (python scripts/download_ved.py --weeks N),
cleans them and writes dashboard/assets/eda_summary.json (a few hundred KB),
which is committed so the page works without the raw data.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from analyzer.ingestion.cleaner import add_vehicle_info, clean_ved  # noqa: E402
from analyzer.ingestion.loader import list_ved_files, load_ved, load_vehicle_info  # noqa: E402
from dashboard.services.eda_service import EDA_PATH, build_eda_summary, save_eda_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=4, help="number of weekly files to summarize")
    args = parser.parse_args()

    files = list_ved_files()[: args.weeks]
    if not files:
        sys.exit("No VED files found. Run `python scripts/download_ved.py --weeks 4` first.")
    print(f"Loading {len(files)} week(s) ...")
    raw = load_ved(weeks=args.weeks)
    clean = add_vehicle_info(clean_ved(raw), load_vehicle_info())
    print(f"  {len(raw):,} raw / {len(clean):,} clean readings")

    summary = build_eda_summary(raw, clean, weeks=[f.name[4:10] for f in files])
    path = save_eda_summary(summary)
    print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
