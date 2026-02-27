"""Parse batch results into per-line labeled JSONL.

Reads data/batches/batch_results.jsonl, maps window-relative line numbers
back to absolute positions, and writes:
  - data/labels/{Dataset}.jsonl  (per-dataset)
  - data/labels/all_labeled.jsonl (combined)

Usage:
    uv run python -m scripts.labeling.parse_labels
"""

import json
import re
import sys
from collections import defaultdict

from scripts.labeling.config import (
    ALL_LABELED_PATH,
    BATCH_RESULTS_PATH,
    DATASETS,
    LABELS_DIR,
    LOGHUB_DIR,
    WINDOW_SIZE,
)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (possibly with language tag)
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    """Parse 'DatasetName_startLine' into (dataset, start_line)."""
    # Find the last underscore followed by digits
    match = re.match(r"^(.+)_(\d+)$", custom_id)
    if not match:
        raise ValueError(f"Cannot parse custom_id: {custom_id}")
    return match.group(1), int(match.group(2))


def load_raw_lines(dataset: str) -> list[str]:
    path = LOGHUB_DIR / f"{dataset}_2k.log"
    if not path.exists():
        return []
    with open(path, "r", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    if not BATCH_RESULTS_PATH.exists():
        print(f"ERROR: {BATCH_RESULTS_PATH} not found. Run collect_results.py first.")
        sys.exit(1)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load all raw lines
    raw_lines: dict[str, list[str]] = {}
    for name, _ in DATASETS:
        lines = load_raw_lines(name)
        if lines:
            raw_lines[name] = lines

    # Parse batch results
    dataset_labels: dict[str, list[dict]] = defaultdict(list)
    parse_errors = 0
    total_lines_labeled = 0

    with open(BATCH_RESULTS_PATH) as f:
        for line_str in f:
            result = json.loads(line_str)
            custom_id = result["custom_id"]

            if result["result"]["type"] != "succeeded":
                print(f"  WARN: {custom_id} — {result['result']['type']}")
                continue

            message = result["result"]["message"]
            text = message["content"][0]["text"]
            text = strip_code_fences(text)

            try:
                entries = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"  ERROR parsing JSON for {custom_id}: {e}")
                parse_errors += 1
                continue

            dataset, start_line = parse_custom_id(custom_id)
            lines_for_dataset = raw_lines.get(dataset, [])

            for entry in entries:
                window_line = entry["line"]  # 1-based within window
                abs_line = start_line + window_line - 1  # 1-based absolute
                line_idx = abs_line - 1  # 0-based for list indexing

                raw_line = ""
                if 0 <= line_idx < len(lines_for_dataset):
                    raw_line = lines_for_dataset[line_idx]

                labeled = {
                    "system": dataset,
                    "line_number": abs_line,
                    "raw_line": raw_line,
                    "label": entry["label"],
                    "reason": entry.get("reason", ""),
                }
                dataset_labels[dataset].append(labeled)
                total_lines_labeled += 1

    # Write per-dataset files
    for dataset in sorted(dataset_labels.keys()):
        labels = sorted(dataset_labels[dataset], key=lambda x: x["line_number"])
        path = LABELS_DIR / f"{dataset}.jsonl"
        with open(path, "w") as f:
            for entry in labels:
                f.write(json.dumps(entry) + "\n")

    # Write combined file
    all_labels = []
    for dataset in sorted(dataset_labels.keys()):
        all_labels.extend(
            sorted(dataset_labels[dataset], key=lambda x: x["line_number"])
        )
    with open(ALL_LABELED_PATH, "w") as f:
        for entry in all_labels:
            f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\n{'Dataset':20s} {'Lines':>7s} {'LOOK':>7s} {'LOOK%':>7s}")
    print("-" * 45)
    for dataset in sorted(dataset_labels.keys()):
        labels = dataset_labels[dataset]
        n_look = sum(1 for l in labels if l["label"] == "LOOK")
        n_total = len(labels)
        pct = (n_look / n_total * 100) if n_total else 0
        print(f"{dataset:20s} {n_total:7d} {n_look:7d} {pct:6.1f}%")

    print("-" * 45)
    total_look = sum(
        1 for labels in dataset_labels.values() for l in labels if l["label"] == "LOOK"
    )
    pct_total = (total_look / total_lines_labeled * 100) if total_lines_labeled else 0
    print(f"{'TOTAL':20s} {total_lines_labeled:7d} {total_look:7d} {pct_total:6.1f}%")

    if parse_errors:
        print(f"\nParse errors: {parse_errors}")

    print(f"\nPer-dataset files: {LABELS_DIR}/")
    print(f"Combined file:     {ALL_LABELED_PATH}")


if __name__ == "__main__":
    main()
