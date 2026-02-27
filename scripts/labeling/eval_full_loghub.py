"""Evaluate the trained LOOK/SKIP model on full loghub datasets.

For BGL (which has ground truth in column 1: "-" = normal, anything else = anomaly),
computes precision/recall/F1. For all other datasets, reports LOOK/SKIP distribution
and samples of LOOK lines that may be missed errors vs correctly flagged.

Usage:
    uv run --group training python -m scripts.labeling.eval_full_loghub [--max-lines 100000]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from scripts.labeling.config import MODEL_PATH
# Must import custom transformers so joblib can unpickle the model
from scripts.labeling.train_model import NormalizeTransformer, HandcraftedFeatureExtractor  # noqa: F401

# Regex to detect lines that contain error-level keywords
_ERROR_KEYWORD_RE = re.compile(
    r"\b(ERROR|FATAL|CRITICAL|PANIC|FAIL(?:ED|URE)?|EXCEPTION|SEVERE|EMERG(?:ENCY)?)\b",
    re.IGNORECASE,
)
_WARN_KEYWORD_RE = re.compile(
    r"\b(WARN(?:ING)?)\b",
    re.IGNORECASE,
)

LOGHUB_DIR = Path("/Users/vadimsemenov/datasets/loghub")

# Datasets to evaluate: (name, log_path_relative, has_ground_truth, max_lines_override)
EVAL_DATASETS = [
    ("BGL", "BGL/BGL.log", True, None),
    ("HDFS", "HDFS_v1/HDFS.log", False, None),
    ("OpenStack-abnormal", "OpenStack/openstack_abnormal.log", False, None),
    ("OpenStack-normal", "OpenStack/openstack_normal1.log", False, None),
    ("Hadoop", None, False, None),  # multiple files, handled specially
    ("HPC", "HPC/HPC.log", False, None),
    ("Linux", "Linux/Linux.log", False, None),
    ("Mac", "Mac/Mac.log", False, None),
    ("SSH", "SSH/SSH.log", False, None),
    ("Zookeeper", "Zookeeper/Zookeeper.log", False, None),
    ("HealthApp", "HealthApp/HealthApp.log", False, None),
    ("Apache", "Apache/Apache.log", False, None),
    ("Proxifier", "Proxifier/Proxifier.log", False, None),
    ("Android", "Android_v1/Android.log", False, None),
    ("Spark", "Spark/Spark.log", False, None),
]


def load_model():
    from joblib import load
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    return load(MODEL_PATH)


def read_lines(path: Path, max_lines: int) -> list[str]:
    lines = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if len(lines) >= max_lines:
                break
    return lines


def read_hadoop_lines(max_lines: int) -> list[str]:
    """Read Hadoop logs from multiple container files."""
    hadoop_dir = LOGHUB_DIR / "Hadoop"
    lines = []
    for log_file in sorted(hadoop_dir.rglob("*.log")):
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
                if len(lines) >= max_lines:
                    return lines
    return lines


def parse_bgl_ground_truth(lines: list[str]) -> tuple[list[str], list[str]]:
    """Parse BGL lines: first token is label ('-' = normal, else = anomaly).
    Returns (raw_lines_without_label, ground_truth_labels)."""
    raw_lines = []
    gt_labels = []
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) < 2:
            raw_lines.append(line)
            gt_labels.append("SKIP")
            continue
        label_token = parts[0]
        rest = parts[1]
        raw_lines.append(rest)
        gt_labels.append("SKIP" if label_token == "-" else "LOOK")
    return raw_lines, gt_labels


def eval_with_ground_truth(model, raw_lines: list[str], gt_labels: list[str], name: str) -> dict:
    """Evaluate model against ground truth, return metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    gt = np.array(gt_labels)
    preds = model.predict(raw_lines)

    n_gt_look = np.sum(gt == "LOOK")
    n_gt_skip = np.sum(gt == "SKIP")
    n_pred_look = np.sum(preds == "LOOK")
    n_pred_skip = np.sum(preds == "SKIP")

    acc = accuracy_score(gt, preds)
    prec = precision_score(gt, preds, pos_label="LOOK", zero_division=0)
    rec = recall_score(gt, preds, pos_label="LOOK", zero_division=0)
    f1 = f1_score(gt, preds, pos_label="LOOK", zero_division=0)
    cm = confusion_matrix(gt, preds, labels=["LOOK", "SKIP"])

    print(f"\n{'='*70}")
    print(f"  {name} — WITH GROUND TRUTH ({len(raw_lines):,} lines)")
    print(f"{'='*70}")
    print(f"  Ground truth:  {n_gt_look:,} LOOK ({n_gt_look/len(raw_lines)*100:.1f}%),  {n_gt_skip:,} SKIP")
    print(f"  Predictions:   {n_pred_look:,} LOOK ({n_pred_look/len(raw_lines)*100:.1f}%),  {n_pred_skip:,} SKIP")
    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  LOOK Precision: {prec:.4f}  (of lines model flagged LOOK, how many are truly anomalies)")
    print(f"  LOOK Recall:    {rec:.4f}  (of actual anomalies, how many the model caught)")
    print(f"  LOOK F1:        {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted LOOK  Predicted SKIP")
    print(f"  Actual LOOK     {cm[0][0]:>13,}  {cm[0][1]:>13,}")
    print(f"  Actual SKIP     {cm[1][0]:>13,}  {cm[1][1]:>13,}")

    # Show missed errors (actual LOOK, predicted SKIP) — most interesting
    missed_mask = (gt == "LOOK") & (preds == "SKIP")
    missed_indices = np.where(missed_mask)[0]
    if len(missed_indices) > 0:
        print(f"\n  MISSED ERRORS (actual=LOOK, predicted=SKIP): {len(missed_indices):,} lines")
        print(f"  Sample (first 15):")
        for idx in missed_indices[:15]:
            line = raw_lines[idx]
            print(f"    [{idx+1}] {line[:120]}")

    # Show false alarms (actual SKIP, predicted LOOK)
    false_alarm_mask = (gt == "SKIP") & (preds == "LOOK")
    false_alarm_indices = np.where(false_alarm_mask)[0]
    if len(false_alarm_indices) > 0:
        print(f"\n  FALSE ALARMS (actual=SKIP, predicted=LOOK): {len(false_alarm_indices):,} lines")
        print(f"  Sample (first 10):")
        for idx in false_alarm_indices[:10]:
            line = raw_lines[idx]
            print(f"    [{idx+1}] {line[:120]}")

    return {
        "n_lines": len(raw_lines),
        "n_gt_look": int(n_gt_look),
        "n_pred_look": int(n_pred_look),
        "accuracy": round(float(acc), 4),
        "look_precision": round(float(prec), 4),
        "look_recall": round(float(rec), 4),
        "look_f1": round(float(f1), 4),
        "missed_errors": int(len(missed_indices)),
        "false_alarms": int(len(false_alarm_indices)),
    }


def eval_no_ground_truth(model, raw_lines: list[str], name: str) -> dict:
    """Predict on dataset without ground truth, show distribution + samples."""
    preds = model.predict(raw_lines)
    n_look = np.sum(preds == "LOOK")
    n_skip = np.sum(preds == "SKIP")
    pct_look = n_look / len(preds) * 100

    print(f"\n  {name:25s}  {len(raw_lines):>9,} lines  →  {n_look:>8,} LOOK ({pct_look:5.1f}%)  {n_skip:>8,} SKIP")

    # Show sample LOOK lines
    look_indices = np.where(preds == "LOOK")[0]
    if len(look_indices) > 0:
        # Sample evenly across the LOOK lines
        sample_indices = look_indices[np.linspace(0, len(look_indices)-1, min(8, len(look_indices)), dtype=int)]
        print(f"    Sample LOOK lines:")
        for idx in sample_indices:
            print(f"      [{idx+1}] {raw_lines[idx][:130]}")

    return {
        "n_lines": len(raw_lines),
        "n_look": int(n_look),
        "n_skip": int(n_skip),
        "pct_look": round(pct_look, 2),
    }


def analyze_keyword_capture(raw_lines: list[str], preds: np.ndarray, name: str) -> dict:
    """Check whether lines with ERROR/FATAL/WARN keywords are classified as LOOK."""
    error_indices = [i for i, line in enumerate(raw_lines) if _ERROR_KEYWORD_RE.search(line)]
    warn_indices = [i for i, line in enumerate(raw_lines) if _WARN_KEYWORD_RE.search(line)]

    result = {}

    if error_indices:
        error_preds = preds[error_indices]
        n_err = len(error_indices)
        n_err_look = int(np.sum(error_preds == "LOOK"))
        n_err_skip = n_err - n_err_look
        capture_rate = n_err_look / n_err * 100

        result["error_lines"] = n_err
        result["error_captured"] = n_err_look
        result["error_missed"] = n_err_skip
        result["error_capture_rate"] = round(capture_rate, 1)

        # Show missed ERROR lines
        missed = [i for i in error_indices if preds[i] == "SKIP"]
        if missed:
            result["missed_error_samples"] = [raw_lines[i][:150] for i in missed[:10]]

    if warn_indices:
        warn_preds = preds[warn_indices]
        n_warn = len(warn_indices)
        n_warn_look = int(np.sum(warn_preds == "LOOK"))
        n_warn_skip = n_warn - n_warn_look
        capture_rate = n_warn_look / n_warn * 100

        result["warn_lines"] = n_warn
        result["warn_captured"] = n_warn_look
        result["warn_missed"] = n_warn_skip
        result["warn_capture_rate"] = round(capture_rate, 1)

    return result


def print_keyword_analysis(kw: dict, name: str) -> None:
    """Print keyword capture analysis."""
    if not kw:
        return

    if "error_lines" in kw:
        print(f"\n  ERROR/FATAL keyword lines: {kw['error_lines']:,}")
        print(f"    Captured (LOOK): {kw['error_captured']:,} ({kw['error_capture_rate']:.1f}%)")
        print(f"    Missed   (SKIP): {kw['error_missed']:,}")
        if kw.get("missed_error_samples"):
            print(f"    Missed samples:")
            for s in kw["missed_error_samples"]:
                print(f"      {s}")

    if "warn_lines" in kw:
        print(f"  WARN keyword lines: {kw['warn_lines']:,}")
        print(f"    Captured (LOOK): {kw['warn_captured']:,} ({kw['warn_capture_rate']:.1f}%)")
        print(f"    Missed   (SKIP): {kw['warn_missed']:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=200000,
                        help="Max lines to read per dataset (default: 200000)")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model()
    print(f"Max lines per dataset: {args.max_lines:,}\n")

    results = {}

    # Process datasets
    no_gt_results = []
    for name, rel_path, has_gt, max_override in EVAL_DATASETS:
        max_lines = max_override or args.max_lines

        # Load lines
        t0 = time.time()
        if name == "Hadoop":
            raw_lines = read_hadoop_lines(max_lines)
        else:
            path = LOGHUB_DIR / rel_path
            if not path.exists():
                print(f"  SKIP {name}: {path} not found")
                continue
            raw_lines = read_lines(path, max_lines)

        if not raw_lines:
            continue

        load_time = time.time() - t0

        if has_gt and name == "BGL":
            raw_lines_clean, gt_labels = parse_bgl_ground_truth(raw_lines)
            t0 = time.time()
            metrics = eval_with_ground_truth(model, raw_lines_clean, gt_labels, name)
            pred_time = time.time() - t0
            metrics["load_time_s"] = round(load_time, 2)
            metrics["predict_time_s"] = round(pred_time, 2)
            # Keyword capture analysis
            preds = model.predict(raw_lines_clean)
            kw = analyze_keyword_capture(raw_lines_clean, preds, name)
            print_keyword_analysis(kw, name)
            metrics["keyword_analysis"] = kw
            results[name] = metrics
        else:
            t0 = time.time()
            preds = model.predict(raw_lines)
            pred_time = time.time() - t0
            preds_arr = np.array(preds)
            n_look = int(np.sum(preds_arr == "LOOK"))
            n_skip = int(np.sum(preds_arr == "SKIP"))
            pct_look = n_look / len(preds_arr) * 100

            print(f"\n  {name:25s}  {len(raw_lines):>9,} lines  →  {n_look:>8,} LOOK ({pct_look:5.1f}%)  {n_skip:>8,} SKIP")

            # Keyword capture analysis
            kw = analyze_keyword_capture(raw_lines, preds_arr, name)
            print_keyword_analysis(kw, name)

            metrics = {
                "n_lines": len(raw_lines),
                "n_look": n_look,
                "n_skip": n_skip,
                "pct_look": round(pct_look, 2),
                "keyword_analysis": kw,
            }
            metrics["load_time_s"] = round(load_time, 2)
            metrics["predict_time_s"] = round(pred_time, 2)
            no_gt_results.append((name, metrics))
            results[name] = metrics

    # Summary table for non-ground-truth datasets
    if no_gt_results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY — All datasets (no ground truth)")
        print(f"{'='*70}")
        print(f"  {'Dataset':25s} {'Lines':>9s} {'LOOK':>8s} {'LOOK%':>6s} {'Predict(s)':>10s}")
        print(f"  {'-'*62}")
        total_lines = 0
        total_look = 0
        for name, m in no_gt_results:
            total_lines += m["n_lines"]
            total_look += m["n_look"]
            print(f"  {name:25s} {m['n_lines']:>9,} {m['n_look']:>8,} {m['pct_look']:>5.1f}% {m['predict_time_s']:>9.1f}s")
        print(f"  {'-'*62}")
        pct = total_look / total_lines * 100 if total_lines else 0
        print(f"  {'TOTAL':25s} {total_lines:>9,} {total_look:>8,} {pct:>5.1f}%")

    # Global keyword capture summary
    print(f"\n{'='*70}")
    print(f"  KEYWORD CAPTURE SUMMARY (ERROR/FATAL/CRITICAL lines → LOOK?)")
    print(f"{'='*70}")
    print(f"  {'Dataset':25s} {'ERROR lines':>11s} {'Captured':>9s} {'Rate':>6s} {'WARN lines':>11s} {'Captured':>9s} {'Rate':>6s}")
    print(f"  {'-'*82}")
    tot_err = tot_err_cap = tot_warn = tot_warn_cap = 0
    for name, m in list(results.items()):
        kw = m.get("keyword_analysis", {})
        err = kw.get("error_lines", 0)
        err_cap = kw.get("error_captured", 0)
        warn = kw.get("warn_lines", 0)
        warn_cap = kw.get("warn_captured", 0)
        tot_err += err
        tot_err_cap += err_cap
        tot_warn += warn
        tot_warn_cap += warn_cap
        err_rate = f"{err_cap/err*100:.1f}%" if err else "—"
        warn_rate = f"{warn_cap/warn*100:.1f}%" if warn else "—"
        print(f"  {name:25s} {err:>11,} {err_cap:>9,} {err_rate:>6s} {warn:>11,} {warn_cap:>9,} {warn_rate:>6s}")
    print(f"  {'-'*82}")
    tot_err_rate = f"{tot_err_cap/tot_err*100:.1f}%" if tot_err else "—"
    tot_warn_rate = f"{tot_warn_cap/tot_warn*100:.1f}%" if tot_warn else "—"
    print(f"  {'TOTAL':25s} {tot_err:>11,} {tot_err_cap:>9,} {tot_err_rate:>6s} {tot_warn:>11,} {tot_warn_cap:>9,} {tot_warn_rate:>6s}")

    # Save results
    report_path = Path("data/models/eval_full_loghub.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
