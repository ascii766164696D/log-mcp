"""Incremental labeling: prepare → submit → collect → parse for new datasets only.

Detects which datasets in config.DATASETS don't yet have labels in
data/labels/all_labeled.jsonl, runs the batch labeling pipeline for only those,
and appends results to all_labeled.jsonl.

Usage:
    uv run --group labeling python -m scripts.labeling.label_new
    uv run --group labeling python -m scripts.labeling.label_new --dry-run  # show what would be labeled
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict

from scripts.labeling.config import (
    ALL_LABELED_PATH,
    BATCHES_DIR,
    DATASETS,
    LABELS_DIR,
    LOGHUB_DIR,
    MODEL,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    WINDOW_SIZE,
)


# ---------------------------------------------------------------------------
# Detect which datasets already have labels
# ---------------------------------------------------------------------------

def get_labeled_datasets() -> set[str]:
    """Return set of dataset names that already exist in all_labeled.jsonl."""
    labeled: set[str] = set()
    if not ALL_LABELED_PATH.exists():
        return labeled
    with open(ALL_LABELED_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line)
                labeled.add(entry["system"])
            except (json.JSONDecodeError, KeyError):
                continue
    return labeled


def get_new_datasets() -> list[tuple[str, bool]]:
    """Return datasets from config that aren't yet labeled."""
    labeled = get_labeled_datasets()
    new = []
    for name, has_gt in DATASETS:
        if name not in labeled:
            log_path = LOGHUB_DIR / f"{name}_2k.log"
            if log_path.exists():
                new.append((name, has_gt))
            else:
                print(f"  SKIP {name}: log file not found at {log_path}")
    return new


# ---------------------------------------------------------------------------
# Batch request building (same logic as prepare_batches.py, scoped to subset)
# ---------------------------------------------------------------------------

def build_requests(datasets: list[tuple[str, bool]]) -> list[dict]:
    """Build Batch API requests for the given datasets."""
    requests: list[dict] = []
    for name, _ in datasets:
        path = LOGHUB_DIR / f"{name}_2k.log"
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        for start in range(0, total, WINDOW_SIZE):
            window = lines[start : start + WINDOW_SIZE]
            end_line = start + len(window)
            numbered = "".join(f"{i}: {line}" for i, line in enumerate(window, start=1))
            user_content = USER_PROMPT_TEMPLATE.format(
                dataset=name,
                start_line=start + 1,
                end_line=end_line,
                total_lines=total,
                lines=numbered,
            )
            requests.append({
                "custom_id": f"{name}_{start + 1}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 4096,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_content}],
                },
            })
    return requests


# ---------------------------------------------------------------------------
# Submit, collect, parse (same logic as individual scripts, self-contained)
# ---------------------------------------------------------------------------

def submit_and_collect(requests: list[dict]) -> list[dict]:
    """Submit batch, poll until done, return results."""
    try:
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
    except ImportError:
        print("ERROR: anthropic package required. Install with:")
        print('  uv pip install "anthropic>=0.42.0"')
        sys.exit(1)

    # Convert to Batch API format
    batch_requests = []
    for raw in requests:
        params = raw["params"]
        batch_requests.append(
            Request(
                custom_id=raw["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model=params["model"],
                    max_tokens=params["max_tokens"],
                    system=params["system"],
                    messages=params["messages"],
                ),
            )
        )

    client = anthropic.Anthropic()
    print(f"  Submitting batch ({len(batch_requests)} requests) ...")
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  Batch ID: {batch.id}")

    # Save incremental manifest
    manifest_path = BATCHES_DIR / "manifest_incremental.json"
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"batch_id": batch.id, "num_requests": len(batch_requests)}, f, indent=2)

    # Poll until done
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"  Status: {batch.processing_status} | "
            f"processing: {counts.processing}, "
            f"succeeded: {counts.succeeded}, "
            f"errored: {counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    # Collect results
    results = []
    for result in client.messages.batches.results(batch.id):
        results.append(json.loads(result.model_dump_json()))

    # Save raw results for debugging
    results_path = BATCHES_DIR / "batch_results_incremental.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Raw results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Parse results → labeled JSONL
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def parse_results(results: list[dict], datasets: list[tuple[str, bool]]) -> list[dict]:
    """Parse batch results into per-line labeled entries."""
    # Load raw lines for all new datasets
    raw_lines: dict[str, list[str]] = {}
    for name, _ in datasets:
        path = LOGHUB_DIR / f"{name}_2k.log"
        with open(path, "r", errors="replace") as f:
            raw_lines[name] = [line.rstrip("\n") for line in f]

    all_entries: list[dict] = []
    dataset_entries: dict[str, list[dict]] = defaultdict(list)
    parse_errors = 0

    for result in results:
        custom_id = result["custom_id"]
        if result["result"]["type"] != "succeeded":
            print(f"  WARN: {custom_id} — {result['result']['type']}")
            continue

        text = result["result"]["message"]["content"][0]["text"]
        text = strip_code_fences(text)

        try:
            entries = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing JSON for {custom_id}: {e}")
            parse_errors += 1
            continue

        # Parse custom_id: "DatasetName_startLine"
        match = re.match(r"^(.+)_(\d+)$", custom_id)
        if not match:
            print(f"  ERROR: cannot parse custom_id: {custom_id}")
            continue
        dataset, start_line = match.group(1), int(match.group(2))
        ds_lines = raw_lines.get(dataset, [])

        for entry in entries:
            window_line = entry["line"]
            abs_line = start_line + window_line - 1
            line_idx = abs_line - 1

            raw_line = ds_lines[line_idx] if 0 <= line_idx < len(ds_lines) else ""

            labeled = {
                "system": dataset,
                "line_number": abs_line,
                "raw_line": raw_line,
                "label": entry["label"],
                "reason": entry.get("reason", ""),
            }
            all_entries.append(labeled)
            dataset_entries[dataset].append(labeled)

    if parse_errors:
        print(f"  Parse errors: {parse_errors}")

    # Write per-dataset label files
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(dataset_entries.keys()):
        entries = sorted(dataset_entries[dataset], key=lambda x: x["line_number"])
        path = LABELS_DIR / f"{dataset}.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    return all_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental labeling for new datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be labeled, don't submit")
    args = parser.parse_args()

    print("Checking for new (unlabeled) datasets ...\n")
    new_datasets = get_new_datasets()

    if not new_datasets:
        print("All datasets are already labeled. Nothing to do.")
        return

    print(f"Found {len(new_datasets)} new dataset(s) to label:")
    for name, _ in new_datasets:
        path = LOGHUB_DIR / f"{name}_2k.log"
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        n_requests = (n_lines + WINDOW_SIZE - 1) // WINDOW_SIZE
        print(f"  {name:20s}  {n_lines:5d} lines  → {n_requests:3d} batch requests")

    requests = build_requests(new_datasets)
    print(f"\nTotal batch requests: {len(requests)}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit these requests. Exiting.")
        return

    # Submit and collect
    results = submit_and_collect(requests)

    # Parse results
    print("\nParsing results ...")
    new_entries = parse_results(results, new_datasets)

    # Append to all_labeled.jsonl
    print(f"\nAppending {len(new_entries)} entries to {ALL_LABELED_PATH} ...")
    with open(ALL_LABELED_PATH, "a") as f:
        # Sort by dataset, then line number
        new_entries.sort(key=lambda x: (x["system"], x["line_number"]))
        for entry in new_entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    from collections import Counter
    ds_counts = Counter(e["system"] for e in new_entries)
    look_counts = Counter(e["system"] for e in new_entries if e["label"] == "LOOK")

    print(f"\n{'Dataset':20s} {'Lines':>7s} {'LOOK':>7s} {'LOOK%':>7s}")
    print("-" * 45)
    for ds in sorted(ds_counts.keys()):
        n = ds_counts[ds]
        nl = look_counts.get(ds, 0)
        print(f"{ds:20s} {n:7d} {nl:7d} {nl/n*100:6.1f}%")
    total = len(new_entries)
    total_look = sum(look_counts.values())
    print("-" * 45)
    print(f"{'TOTAL':20s} {total:7d} {total_look:7d} {total_look/total*100:6.1f}%")

    # Report final combined total
    with open(ALL_LABELED_PATH) as f:
        combined_total = sum(1 for _ in f)
    print(f"\nCombined all_labeled.jsonl: {combined_total:,} lines")


if __name__ == "__main__":
    main()
