"""Download the Vehicle Energy Dataset (VED) into data/raw/ved/.

Usage:
    python scripts/download_ved.py            # download and extract all weeks (~3 GB)
    python scripts/download_ved.py --weeks 4  # extract only the first 4 weeks
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import py7zr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer.config import (  # noqa: E402
    VED_BASE_URL,
    VED_DIR,
    VED_DYNAMIC_ARCHIVES,
    VED_DYNAMIC_DIR,
    VED_STATIC_DIR,
    VED_STATIC_FILES,
)


def download(url: str, target: Path) -> None:
    """Download `url` to `target`, skipping files that already exist."""
    if target.exists():
        print(f"  already downloaded: {target.name}")
        return
    print(f"  downloading {target.name} ...")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / total:6.1%}", end="", flush=True)
    print()
    tmp.rename(target)


def extract_weeks(archives: list[Path], weeks: int | None) -> None:
    """Extract weekly CSVs from the 7z archives, in date order, up to `weeks` files."""
    names = []
    for archive in archives:
        with py7zr.SevenZipFile(archive) as z:
            names += [(name, archive) for name in z.getnames() if name.endswith("_week.csv")]
    names.sort()
    selected = names[:weeks]
    for archive in archives:
        wanted = [
            name for name, src in selected
            if src == archive and not (VED_DYNAMIC_DIR / name).exists()
        ]
        if wanted:
            print(f"  extracting {len(wanted)} week(s) from {archive.name} ...")
            with py7zr.SevenZipFile(archive) as z:
                z.extract(path=VED_DYNAMIC_DIR, targets=wanted)
    print(f"  {len(selected)} weekly file(s) ready in {VED_DYNAMIC_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=None, help="extract only the first N weeks")
    args = parser.parse_args()

    archive_dir = VED_DIR / "archives"
    for folder in (archive_dir, VED_DYNAMIC_DIR, VED_STATIC_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print("Static vehicle data:")
    for name in VED_STATIC_FILES:
        download(VED_BASE_URL + urllib.request.quote(name), VED_STATIC_DIR / name)

    print("Dynamic driving data:")
    archives = [archive_dir / name for name in VED_DYNAMIC_ARCHIVES]
    for name, path in zip(VED_DYNAMIC_ARCHIVES, archives):
        download(VED_BASE_URL + name, path)
    extract_weeks(archives, args.weeks)


if __name__ == "__main__":
    main()
