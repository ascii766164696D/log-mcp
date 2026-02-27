"""Submit batch requests to the Anthropic Batch API.

Reads requests from data/batches/batch_requests.jsonl, submits them as a
single batch, and saves the batch ID to data/batches/manifest.json.

Usage:
    uv run python -m scripts.labeling.submit_batch
"""

import json
import sys

from scripts.labeling.config import BATCH_MANIFEST_PATH, BATCH_REQUESTS_PATH


def main() -> None:
    try:
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
    except ImportError:
        print("ERROR: anthropic package required. Install with:")
        print('  uv pip install "anthropic>=0.42.0"')
        sys.exit(1)

    if not BATCH_REQUESTS_PATH.exists():
        print(f"ERROR: {BATCH_REQUESTS_PATH} not found. Run prepare_batches.py first.")
        sys.exit(1)

    # Load requests
    raw_requests = []
    with open(BATCH_REQUESTS_PATH) as f:
        for line in f:
            raw_requests.append(json.loads(line))

    print(f"Loaded {len(raw_requests)} requests")

    # Convert to Batch API format
    batch_requests = []
    for raw in raw_requests:
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

    print("Submitting batch...")
    batch = client.messages.batches.create(requests=batch_requests)

    print(f"Batch ID:   {batch.id}")
    print(f"Status:     {batch.processing_status}")
    print(f"Requests:   {len(batch_requests)}")

    # Save manifest
    BATCH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "batch_id": batch.id,
        "processing_status": batch.processing_status,
        "num_requests": len(batch_requests),
    }
    with open(BATCH_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {BATCH_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
