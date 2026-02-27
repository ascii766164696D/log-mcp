"""Validate Claude's labels against BGL/Thunderbird ground truth.

For BGL and Thunderbird, the structured CSV has a "Label" column where "-"
means normal (SKIP) and anything else means anomaly (LOOK).

Computes precision, recall, F1, accuracy, and prints a confusion matrix
plus a sample of disagreements.

Usage:
    uv run python -m scripts.labeling.validate_labels
"""

import csv
import json
import sys

from scripts.labeling.config import LABELS_DIR, LOGHUB_DIR


GROUND_TRUTH_DATASETS = ["BGL", "Thunderbird"]


def load_ground_truth(dataset: str) -> dict[int, str]:
    """Load ground truth: line_number -> 'LOOK' or 'SKIP'."""
    csv_path = LOGHUB_DIR / f"{dataset}_2k.log_structured.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return {}

    truth: dict[int, str] = {}
    with open(csv_path, "r", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            label_val = row.get("Label", "-")
            truth[i] = "SKIP" if label_val == "-" else "LOOK"
    return truth


def load_claude_labels(dataset: str) -> dict[int, str]:
    """Load Claude labels: line_number -> 'LOOK' or 'SKIP'."""
    path = LABELS_DIR / f"{dataset}.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}

    labels: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            labels[entry["line_number"]] = entry["label"]
    return labels


def compute_metrics(
    truth: dict[int, str], predicted: dict[int, str]
) -> dict:
    """Compute classification metrics. Positive = LOOK."""
    tp = fp = fn = tn = 0
    disagreements: list[dict] = []

    for line_num in sorted(truth.keys()):
        if line_num not in predicted:
            continue
        t = truth[line_num]
        p = predicted[line_num]

        if t == "LOOK" and p == "LOOK":
            tp += 1
        elif t == "SKIP" and p == "LOOK":
            fp += 1
            disagreements.append({"line": line_num, "truth": t, "predicted": p})
        elif t == "LOOK" and p == "SKIP":
            fn += 1
            disagreements.append({"line": line_num, "truth": t, "predicted": p})
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "disagreements": disagreements,
    }


def main() -> None:
    any_found = False

    for dataset in GROUND_TRUTH_DATASETS:
        print(f"\n{'='*60}")
        print(f"  {dataset}")
        print(f"{'='*60}")

        truth = load_ground_truth(dataset)
        predicted = load_claude_labels(dataset)

        if not truth:
            print("  No ground truth available.")
            continue
        if not predicted:
            print("  No Claude labels available.")
            continue

        any_found = True
        overlap = set(truth.keys()) & set(predicted.keys())
        print(f"  Ground truth lines: {len(truth)}")
        print(f"  Claude-labeled lines: {len(predicted)}")
        print(f"  Overlap: {len(overlap)}")

        metrics = compute_metrics(truth, predicted)

        # Confusion matrix
        print(f"\n  Confusion Matrix (positive = LOOK):")
        print(f"                  Predicted LOOK  Predicted SKIP")
        print(f"  Actual LOOK     {metrics['tp']:>14d}  {metrics['fn']:>14d}")
        print(f"  Actual SKIP     {metrics['fp']:>14d}  {metrics['tn']:>14d}")

        print(f"\n  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")

        # Sample disagreements
        disag = metrics["disagreements"]
        if disag:
            n_show = min(10, len(disag))
            print(f"\n  Sample disagreements ({n_show} of {len(disag)}):")
            # Load raw lines for context
            log_path = LOGHUB_DIR / f"{dataset}_2k.log"
            raw_lines: list[str] = []
            if log_path.exists():
                with open(log_path, "r", errors="replace") as f:
                    raw_lines = [l.rstrip("\n") for l in f]

            for d in disag[:n_show]:
                line_idx = d["line"] - 1
                raw = raw_lines[line_idx][:120] if line_idx < len(raw_lines) else "?"
                print(f"    L{d['line']:5d} truth={d['truth']:4s} pred={d['predicted']:4s}  {raw}")

    if not any_found:
        print("\nNo ground truth datasets found. Run download and labeling first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
