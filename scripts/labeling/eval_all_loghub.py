"""Run LOOK/SKIP model on full loghub datasets, report per-dataset stats.

Pre-split large files live in _splits/. Each file is a work unit processed
by one worker. Uses fork-based multiprocessing.

Usage:
    uv run --group training python -m scripts.labeling.eval_all_loghub
    uv run --group training python -m scripts.labeling.eval_all_loghub --workers 16
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.labeling.config import MODEL_PATH
from scripts.labeling.train_model import NormalizeTransformer, HandcraftedFeatureExtractor  # noqa: F401

LOGHUB_DIR = Path("/Users/vadimsemenov/datasets/loghub")
SPLITS_DIR = LOGHUB_DIR / "_splits"
CHUNK_SIZE = 50_000  # lines per model.predict() call

_ERROR_RE = re.compile(
    r"\b(ERROR|FATAL|CRITICAL|PANIC|FAIL(?:ED|URE)?|EXCEPTION|SEVERE|EMERG(?:ENCY)?)\b",
    re.IGNORECASE,
)
_WARN_RE = re.compile(r"\b(WARN(?:ING)?)\b", re.IGNORECASE)


def build_work_units() -> list[tuple[str, str]]:
    """Build list of (dataset_name, file_path) work units.

    For large datasets that were pre-split into _splits/, use those chunks.
    For everything else, use the original file.
    """
    units: list[tuple[str, str]] = []

    # Datasets with pre-split files
    split_datasets = {"Thunderbird", "Windows", "HDFS", "BGL"}

    # Small/medium datasets — use original files
    originals = [
        ("Android", "Android_v1/Android.log"),
        ("Apache", "Apache/Apache.log"),
        ("HealthApp", "HealthApp/HealthApp.log"),
        ("HPC", "HPC/HPC.log"),
        ("Linux", "Linux/Linux.log"),
        ("Mac", "Mac/Mac.log"),
        ("OpenStack-abnormal", "OpenStack/openstack_abnormal.log"),
        ("OpenStack-normal", "OpenStack/openstack_normal1.log"),
        ("OpenStack-normal2", "OpenStack/openstack_normal2.log"),
        ("Proxifier", "Proxifier/Proxifier.log"),
        ("SSH", "SSH/SSH.log"),
        ("Spark", "Spark/Spark.log"),
        ("Zookeeper", "Zookeeper/Zookeeper.log"),
    ]
    for name, rel in originals:
        p = LOGHUB_DIR / rel
        if p.exists():
            units.append((name, str(p)))

    # Hadoop — many small files in subdirectories
    hadoop_dir = LOGHUB_DIR / "Hadoop"
    if hadoop_dir.exists():
        for f in sorted(hadoop_dir.rglob("*.log")):
            units.append(("Hadoop", str(f)))

    # Pre-split large files
    if SPLITS_DIR.exists():
        for name in sorted(split_datasets):
            prefix = name + "_"
            chunks = sorted(SPLITS_DIR.glob(prefix + "*"))
            for c in chunks:
                units.append((name, str(c)))

    return units


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_worker_model = None


def _worker_init(model_path: str):
    global _worker_model
    from joblib import load
    _worker_model = load(model_path)


def _process_file(args: tuple[str, str]) -> dict:
    """Process a single file: read all lines, predict in chunks, count keywords."""
    dataset, file_path = args
    model = _worker_model

    total = 0
    n_look = 0
    n_error_kw = 0
    n_warn_kw = 0
    n_error_captured = 0
    n_error_missed = 0
    t0 = time.time()

    chunk: list[str] = []

    with open(file_path, "r", errors="replace") as f:
        for line in f:
            chunk.append(line.rstrip("\n"))
            if len(chunk) >= CHUNK_SIZE:
                preds = np.array(model.predict(chunk))
                total += len(chunk)
                n_look += int(np.sum(preds == "LOOK"))

                err_mask = np.array([bool(_ERROR_RE.search(l)) for l in chunk])
                warn_mask = np.array([bool(_WARN_RE.search(l)) for l in chunk])
                n_error_kw += int(err_mask.sum())
                n_warn_kw += int(warn_mask.sum())
                n_error_captured += int(np.sum(err_mask & (preds == "LOOK")))
                n_error_missed += int(np.sum(err_mask & (preds == "SKIP")))
                chunk = []

    # Remaining
    if chunk:
        preds = np.array(model.predict(chunk))
        total += len(chunk)
        n_look += int(np.sum(preds == "LOOK"))
        err_mask = np.array([bool(_ERROR_RE.search(l)) for l in chunk])
        warn_mask = np.array([bool(_WARN_RE.search(l)) for l in chunk])
        n_error_kw += int(err_mask.sum())
        n_warn_kw += int(warn_mask.sum())
        n_error_captured += int(np.sum(err_mask & (preds == "LOOK")))
        n_error_missed += int(np.sum(err_mask & (preds == "SKIP")))

    elapsed = time.time() - t0
    fname = os.path.basename(file_path)
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  {dataset:20s} {fname:30s} {total:>12,} lines  {elapsed:>6.1f}s  ({rate:,.0f} l/s)", flush=True)

    return {
        "dataset": dataset,
        "total": total,
        "look": n_look,
        "error_kw": n_error_kw,
        "warn_kw": n_warn_kw,
        "error_captured": n_error_captured,
        "error_missed": n_error_missed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 16))
    args = parser.parse_args()
    n_workers = args.workers

    print(f"Model: {MODEL_PATH}")
    print(f"Workers: {n_workers}")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    units = build_work_units()
    ds_files = defaultdict(int)
    for ds, _ in units:
        ds_files[ds] += 1

    print(f"\nWork units: {len(units)} files across {len(ds_files)} datasets")
    for ds in sorted(ds_files):
        print(f"  {ds:25s}  {ds_files[ds]:>3d} file(s)")

    # Sort by file size descending for better load balancing
    units.sort(key=lambda u: os.path.getsize(u[1]), reverse=True)

    print(f"\nProcessing ...\n")
    t0 = time.time()

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(str(MODEL_PATH),)) as pool:
        results = pool.map(_process_file, units, chunksize=1)

    wall = time.time() - t0

    # Aggregate by dataset
    agg: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "look": 0, "error_kw": 0, "warn_kw": 0,
        "error_captured": 0, "error_missed": 0,
    })
    for r in results:
        ds = r["dataset"]
        for k in ["total", "look", "error_kw", "warn_kw", "error_captured", "error_missed"]:
            agg[ds][k] += r[k]

    # Display order
    display_order = [
        "Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC",
        "Linux", "Mac", "OpenStack-abnormal", "OpenStack-normal", "OpenStack-normal2",
        "Proxifier", "SSH", "Spark", "Thunderbird", "Windows", "Zookeeper",
    ]
    ordered = [n for n in display_order if n in agg]

    print(f"\n{'='*120}")
    print(f"  {'Dataset':25s} {'Total':>12s} {'LOOK':>12s} {'LOOK%':>6s} {'ERROR kw':>10s} {'ERR→LOOK':>10s} {'ERR→SKIP':>10s} {'ERR cap%':>8s} {'WARN kw':>10s}")
    print(f"  {'-'*107}")

    g = defaultdict(int)
    for name in ordered:
        r = agg[name]
        pct = r["look"] / r["total"] * 100 if r["total"] else 0
        ecap = f"{r['error_captured']/r['error_kw']*100:.1f}%" if r["error_kw"] else "—"
        print(
            f"  {name:25s} {r['total']:>12,} {r['look']:>12,} {pct:>5.1f}%"
            f" {r['error_kw']:>10,} {r['error_captured']:>10,} {r['error_missed']:>10,} {ecap:>8s}"
            f" {r['warn_kw']:>10,}"
        )
        for k in ["total", "look", "error_kw", "error_captured", "error_missed", "warn_kw"]:
            g[k] += r[k]

    print(f"  {'-'*107}")
    gpct = g["look"] / g["total"] * 100 if g["total"] else 0
    gecap = f"{g['error_captured']/g['error_kw']*100:.1f}%" if g["error_kw"] else "—"
    print(
        f"  {'TOTAL':25s} {g['total']:>12,} {g['look']:>12,} {gpct:>5.1f}%"
        f" {g['error_kw']:>10,} {g['error_captured']:>10,} {g['error_missed']:>10,} {gecap:>8s}"
        f" {g['warn_kw']:>10,}"
    )
    rate = g["total"] / wall if wall else 0
    print(f"\n  Wall time: {wall:.1f}s ({rate:,.0f} lines/s effective)")


if __name__ == "__main__":
    main()
