#!/usr/bin/env python3
"""Evaluate Rust LOOK/SKIP classifier on all loghub datasets.

Single-pass evaluation: classify_file_with_keywords does classification +
keyword analysis in one parallel Rust pass using Rayon.
"""

import json
import os
import time

from look_skip_classifier import LookSkipClassifier

LOGHUB = "/Users/vadimsemenov/datasets/loghub"
MODEL_PATH = "data/models/look_skip_model.json"
PREV_EVAL_PATH = "data/models/eval_full_loghub.json"
OUTPUT_PATH = "data/models/eval_rust_loghub.json"

DATASETS = {
    "BGL": "BGL/BGL.log",
    "HDFS": "HDFS_v1/HDFS.log",
    "OpenStack-abnormal": "OpenStack/openstack_abnormal.log",
    "OpenStack-normal": "OpenStack/openstack_normal1.log",
    "Hadoop": "Hadoop/",
    "HPC": "HPC/HPC.log",
    "Linux": "Linux/Linux.log",
    "Mac": "Mac/Mac.log",
    "SSH": "SSH/SSH.log",
    "Zookeeper": "Zookeeper/Zookeeper.log",
    "HealthApp": "HealthApp/HealthApp.log",
    "Apache": "Apache/Apache.log",
    "Proxifier": "Proxifier/Proxifier.log",
    "Android": "Android_v1/",
    "Thunderbird": "Thunderbird/Thunderbird.log",
    "Windows": "Windows/Windows.log",
    "Spark": "Spark/",
    "HDFS_v2": "HDFS_v2/",
}


def log(msg: str):
    print(msg, flush=True)


def find_log_files(path: str) -> list[str]:
    full = os.path.join(LOGHUB, path)
    if os.path.isfile(full):
        return [full]
    if os.path.isdir(full):
        logs = []
        for root, _, files in os.walk(full):
            for f in sorted(files):
                if f.endswith(".log"):
                    logs.append(os.path.join(root, f))
        return logs
    return []


def eval_dataset(clf: LookSkipClassifier, name: str, path: str) -> dict:
    files = find_log_files(path)
    if not files:
        log(f"  SKIP {name}: no log files found")
        return {}

    total_lines = 0
    total_look = 0
    total_skip = 0
    total_time = 0.0
    error_lines = 0
    error_captured = 0
    warn_lines = 0
    warn_captured = 0
    sample_look = []

    for fp in files:
        # Single Rust pass: classify + keyword analysis
        r = clf.classify_file_with_keywords(fp, 0.5, 0, 500)
        total_lines += r["total_lines"]
        total_look += r["look_count"]
        total_skip += r["skip_count"]
        total_time += r["processing_time_s"]
        error_lines += r["error_lines"]
        error_captured += r["error_captured"]
        warn_lines += r["warn_lines"]
        warn_captured += r["warn_captured"]

        if len(sample_look) < 5:
            for ln, prob, text in r["look_lines"][:5 - len(sample_look)]:
                sample_look.append((ln, prob, text[:120]))

    pct_look = (total_look / total_lines * 100) if total_lines > 0 else 0
    rate = total_lines / total_time if total_time > 0 else 0
    err_rate = (error_captured / error_lines * 100) if error_lines > 0 else 100.0
    warn_rate = (warn_captured / warn_lines * 100) if warn_lines > 0 else 100.0

    log(f"  {total_lines:>12,} lines in {total_time:>6.1f}s = {rate:>10,.0f} lines/sec | LOOK: {total_look:>10,} ({pct_look:.1f}%)")
    log(f"  Errors: {error_lines:>8,} ({err_rate:.1f}% captured) | Warns: {warn_lines:>8,} ({warn_rate:.1f}% captured) | {len(files)} file(s)")

    return {
        "n_lines": total_lines,
        "n_look": total_look,
        "n_skip": total_skip,
        "pct_look": round(pct_look, 2),
        "classify_time_s": round(total_time, 2),
        "lines_per_second": round(rate),
        "keyword_analysis": {
            "error_lines": error_lines,
            "error_captured": error_captured,
            "error_missed": error_lines - error_captured,
            "error_capture_rate": round(err_rate, 2),
            "warn_lines": warn_lines,
            "warn_captured": warn_captured,
            "warn_missed": warn_lines - warn_captured,
            "warn_capture_rate": round(warn_rate, 2),
        },
        "sample_look": [
            {"line": ln, "prob": round(prob, 4), "text": text}
            for ln, prob, text in sample_look
        ],
        "n_files": len(files),
    }


def main():
    log(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    clf = LookSkipClassifier(MODEL_PATH)
    load_time = time.time() - t0
    log(f"Model loaded in {load_time:.2f}s\n")

    prev = {}
    if os.path.exists(PREV_EVAL_PATH):
        with open(PREV_EVAL_PATH) as f:
            prev = json.load(f)

    results = {}
    grand_lines = 0
    grand_time = 0.0
    grand_errors = 0
    grand_errors_cap = 0
    grand_warns = 0
    grand_warns_cap = 0

    wall_start = time.time()

    for name, path in DATASETS.items():
        log(f"--- {name} ---")
        result = eval_dataset(clf, name, path)
        if not result:
            continue

        results[name] = result
        grand_lines += result["n_lines"]
        grand_time += result["classify_time_s"]
        kw = result["keyword_analysis"]
        grand_errors += kw["error_lines"]
        grand_errors_cap += kw["error_captured"]
        grand_warns += kw["warn_lines"]
        grand_warns_cap += kw["warn_captured"]

        if name in prev:
            pt = prev[name].get("predict_time_s", prev[name].get("classify_time_s", 0))
            if pt > 0 and result["classify_time_s"] > 0:
                log(f"  vs Python: {pt:.1f}s -> {result['classify_time_s']:.1f}s ({pt/result['classify_time_s']:.0f}x)")
        log("")

    wall_time = time.time() - wall_start
    err_rate = (grand_errors_cap / grand_errors * 100) if grand_errors > 0 else 100.0
    warn_rate = (grand_warns_cap / grand_warns * 100) if grand_warns > 0 else 100.0

    log(f"{'='*60}")
    log(f"GRAND TOTAL")
    log(f"{'='*60}")
    log(f"  Lines:     {grand_lines:>14,}")
    log(f"  Time:      {grand_time:>12.1f}s classify / {wall_time:.1f}s wall")
    if grand_time > 0:
        log(f"  Rate:      {grand_lines / grand_time:>12,.0f} lines/sec")
    log(f"  Errors:    {grand_errors:>12,} total, {grand_errors_cap:>10,} captured ({err_rate:.2f}%)")
    log(f"  Warns:     {grand_warns:>12,}  total, {grand_warns_cap:>10,} captured ({warn_rate:.2f}%)")
    log(f"  Missed:    {grand_errors - grand_errors_cap:>12,} errors, {grand_warns - grand_warns_cap:>10,} warns")

    if prev:
        prev_total = sum(v.get("predict_time_s", v.get("classify_time_s", 0)) for v in prev.values())
        if prev_total > 0 and grand_time > 0:
            log(f"\n  Previous Python (200K samples/dataset): {prev_total:.0f}s")
            log(f"  Rust (full files, all lines):            {grand_time:.0f}s")

    output = {"summary": {
        "total_lines": grand_lines,
        "total_classify_time_s": round(grand_time, 2),
        "wall_time_s": round(wall_time, 2),
        "overall_lines_per_second": round(grand_lines / grand_time) if grand_time > 0 else 0,
        "total_errors": grand_errors,
        "total_errors_captured": grand_errors_cap,
        "total_warns": grand_warns,
        "total_warns_captured": grand_warns_cap,
        "error_capture_rate": round(err_rate, 4),
        "model_load_time_s": round(load_time, 2),
    }, "datasets": results}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
