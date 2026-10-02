"""Generate synthetic fault logs from cleaned VED data into data/synthetic/.

Usage:
    python scripts/generate_synthetic.py                       # 120 faulty + 120 normal trips
    python scripts/generate_synthetic.py --faulty 300 --normal 300 --seed 7

Requires data/processed/ved_clean.parquet (run scripts/prepare_ved.py first).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.config import PROCESSED_DIR, SYNTHETIC_DIR  # noqa: E402
from analyzer.synthetic.generator import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faulty", type=int, default=120, help="number of trips with an injected fault")
    parser.add_argument("--normal", type=int, default=120, help="number of normal trips")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same dataset)")
    args = parser.parse_args()

    source = PROCESSED_DIR / "ved_clean.parquet"
    if not source.exists():
        sys.exit(f"{source} not found. Run `python scripts/prepare_ved.py` first.")

    print(f"Loading {source.name} ...")
    clean = pd.read_parquet(source)

    print("Injecting faults ...")
    data = generate(clean, n_faulty=args.faulty, n_normal=args.normal, seed=args.seed)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    data.logs.to_parquet(SYNTHETIC_DIR / "synthetic_logs.parquet", index=False)
    data.dtc_events.to_csv(SYNTHETIC_DIR / "synthetic_dtc_events.csv", index=False)
    data.labels.to_csv(SYNTHETIC_DIR / "synthetic_labels.csv", index=False)

    print(f"  {len(data.labels)} trips, {len(data.logs):,} rows, {len(data.dtc_events)} DTC events")
    print("  trips per fault:")
    for fault, count in data.labels["fault"].value_counts().items():
        print(f"    {fault:24s} {count}")
    print(f"Saved to {SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()
