"""Download 2k-line log samples from the loghub repository.

Fetches {Name}_2k.log for each dataset, plus structured CSVs for
ground-truth datasets (BGL, Thunderbird). Idempotent — skips existing files.

Usage:
    uv run python -m scripts.labeling.download_loghub
"""

import sys
from pathlib import Path
import urllib.request
import urllib.error

from scripts.labeling.config import DATASETS, LOGHUB_BASE_URL, LOGHUB_DIR


def download_file(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped."""
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return False
    print(f"  downloading:   {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("ok")
        return True
    except urllib.error.HTTPError as e:
        print(f"FAILED ({e.code} {e.reason})")
        return False


def main() -> None:
    LOGHUB_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    failures = []

    for name, has_ground_truth in DATASETS:
        print(f"\n[{name}]")
        log_filename = f"{name}_2k.log"
        log_url = f"{LOGHUB_BASE_URL}/{name}/{log_filename}"
        log_dest = LOGHUB_DIR / log_filename

        if download_file(log_url, log_dest):
            total_downloaded += 1
        elif log_dest.exists():
            total_skipped += 1
        else:
            failures.append(log_filename)

        if has_ground_truth:
            csv_filename = f"{name}_2k.log_structured.csv"
            csv_url = f"{LOGHUB_BASE_URL}/{name}/{csv_filename}"
            csv_dest = LOGHUB_DIR / csv_filename

            if download_file(csv_url, csv_dest):
                total_downloaded += 1
            elif csv_dest.exists():
                total_skipped += 1
            else:
                failures.append(csv_filename)

    # Verify line counts
    print("\n--- Line counts ---")
    for name, _ in DATASETS:
        log_path = LOGHUB_DIR / f"{name}_2k.log"
        if log_path.exists():
            count = sum(1 for _ in open(log_path, "r", errors="replace"))
            marker = "" if count == 2000 else f"  (expected 2000)"
            print(f"  {name:20s} {count:>5d} lines{marker}")

    print(f"\nDownloaded: {total_downloaded}, Skipped: {total_skipped}")
    if failures:
        print(f"Failures: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
