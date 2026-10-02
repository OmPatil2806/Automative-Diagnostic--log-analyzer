"""Load, clean and save VED logs to data/processed/ved_clean.parquet.

Usage:
    python scripts/prepare_ved.py            # all downloaded weeks
    python scripts/prepare_ved.py --weeks 1  # only the first week
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.config import PROCESSED_DIR  # noqa: E402
from analyzer.ingestion.cleaner import add_vehicle_info, clean_ved, summarize  # noqa: E402
from analyzer.ingestion.loader import load_ved, load_vehicle_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=None, help="process only the first N weeks")
    args = parser.parse_args()

    print("Loading raw logs ...")
    raw = load_ved(weeks=args.weeks)
    print(f"  {len(raw):,} raw rows")

    print("Cleaning ...")
    clean = add_vehicle_info(clean_ved(raw), load_vehicle_info())
    stats = summarize(clean)
    print(f"  {stats['rows']:,} rows | {stats['vehicles']} vehicles | {stats['trips']:,} trips")
    print(f"  {stats['start']:%Y-%m-%d} to {stats['end']:%Y-%m-%d}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "ved_clean.parquet"
    clean.to_parquet(out, index=False)
    print(f"Saved {out.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()
