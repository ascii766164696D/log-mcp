"""Build batch request JSONL from downloaded loghub samples.

Slides a non-overlapping window of WINDOW_SIZE lines over each dataset and
creates one Batch API request per window. Writes all requests to
data/batches/batch_requests.jsonl.

Usage:
    uv run python -m scripts.labeling.prepare_batches
    uv run python -m scripts.labeling.prepare_batches --dry-test  # test one window via Messages API
"""

import argparse
import json
import sys

from scripts.labeling.config import (
    BATCH_REQUESTS_PATH,
    DATASETS,
    LOGHUB_DIR,
    MODEL,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    WINDOW_SIZE,
)


def read_lines(dataset_name: str) -> list[str]:
    path = LOGHUB_DIR / f"{dataset_name}_2k.log"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    with open(path, "r", errors="replace") as f:
        return f.readlines()


def build_request(dataset: str, start_line: int, lines: list[str], total_lines: int) -> dict:
    """Build a single Batch API request dict."""
    end_line = start_line + len(lines) - 1
    numbered_lines = "".join(
        f"{i}: {line}" for i, line in enumerate(lines, start=1)
    )
    user_content = USER_PROMPT_TEMPLATE.format(
        dataset=dataset,
        start_line=start_line,
        end_line=end_line,
        total_lines=total_lines,
        lines=numbered_lines,
    )
    return {
        "custom_id": f"{dataset}_{start_line}",
        "params": {
            "model": MODEL,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-test",
        action="store_true",
        help="Send one window to the Messages API to verify the prompt",
    )
    args = parser.parse_args()

    BATCH_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_requests: list[dict] = []
    for dataset_name, _ in DATASETS:
        lines = read_lines(dataset_name)
        if not lines:
            continue
        total = len(lines)
        for start in range(0, total, WINDOW_SIZE):
            window = lines[start : start + WINDOW_SIZE]
            req = build_request(dataset_name, start + 1, window, total)
            all_requests.append(req)

    print(f"Total requests: {len(all_requests)}")
    print(f"Datasets: {len([d for d, _ in DATASETS if (LOGHUB_DIR / f'{d}_2k.log').exists()])}")

    if args.dry_test:
        _dry_test(all_requests[0])
        return

    with open(BATCH_REQUESTS_PATH, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Written to {BATCH_REQUESTS_PATH}")


def _dry_test(request: dict) -> None:
    """Send a single request via the Messages API to verify the prompt."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package required for dry-test. Install with:")
        print("  uv pip install anthropic")
        sys.exit(1)

    print(f"\n--- Dry test: {request['custom_id']} ---")
    params = request["params"]
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=params["model"],
        max_tokens=params["max_tokens"],
        system=params["system"],
        messages=params["messages"],
    )
    text = response.content[0].text
    print(f"Response ({response.usage.input_tokens}in/{response.usage.output_tokens}out tokens):")
    print(text[:2000])

    # Try to parse
    try:
        parsed = json.loads(text)
        print(f"\nParsed OK: {len(parsed)} entries")
        for entry in parsed[:5]:
            print(f"  line {entry['line']}: {entry['label']} — {entry['reason']}")
    except json.JSONDecodeError as e:
        print(f"\nJSON parse failed: {e}")
        # Try stripping markdown fences
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            try:
                parsed = json.loads(stripped)
                print(f"Parsed after stripping fences: {len(parsed)} entries")
            except json.JSONDecodeError:
                print("Still failed after stripping fences")


if __name__ == "__main__":
    main()
