"""Poll the Batch API until complete and download results.

Reads batch ID from data/batches/manifest.json, polls every 60s,
then streams results to data/batches/batch_results.jsonl.

Usage:
    uv run python -m scripts.labeling.collect_results
"""

import json
import sys
import time

from scripts.labeling.config import BATCH_MANIFEST_PATH, BATCH_RESULTS_PATH


def main() -> None:
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package required. Install with:")
        print('  uv pip install "anthropic>=0.42.0"')
        sys.exit(1)

    if not BATCH_MANIFEST_PATH.exists():
        print(f"ERROR: {BATCH_MANIFEST_PATH} not found. Run submit_batch.py first.")
        sys.exit(1)

    with open(BATCH_MANIFEST_PATH) as f:
        manifest = json.load(f)

    batch_id = manifest["batch_id"]
    print(f"Batch ID: {batch_id}")

    client = anthropic.Anthropic()

    # Poll until ended
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"Status: {batch.processing_status} | "
            f"processing: {counts.processing}, "
            f"succeeded: {counts.succeeded}, "
            f"errored: {counts.errored}, "
            f"expired: {counts.expired}, "
            f"canceled: {counts.canceled}"
        )

        if batch.processing_status == "ended":
            break

        time.sleep(60)

    print(f"\nBatch complete. Succeeded: {counts.succeeded}, Errored: {counts.errored}")

    # Stream results to JSONL
    BATCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_count = 0
    with open(BATCH_RESULTS_PATH, "w") as f:
        for result in client.messages.batches.results(batch_id):
            f.write(result.model_dump_json() + "\n")
            result_count += 1

    print(f"Saved {result_count} results to {BATCH_RESULTS_PATH}")

    # Update manifest
    manifest["processing_status"] = "ended"
    manifest["succeeded"] = counts.succeeded
    manifest["errored"] = counts.errored
    with open(BATCH_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
