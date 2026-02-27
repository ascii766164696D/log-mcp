"""Orchestrate the full labeling pipeline.

Runs all steps sequentially. Each step is also independently runnable
as a standalone module.

Usage:
    uv run python -m scripts.labeling.run_pipeline
    uv run python -m scripts.labeling.run_pipeline --skip-batch  # skip submit/collect (for re-parsing)
"""

import argparse
import sys

from scripts.labeling import (
    download_loghub,
    prepare_batches,
    submit_batch,
    collect_results,
    parse_labels,
    validate_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full labeling pipeline")
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch submission and collection (useful for re-parsing existing results)",
    )
    args = parser.parse_args()

    steps: list[tuple[str, object]] = [
        ("Step 1: Download loghub samples", download_loghub.main),
        ("Step 2: Prepare batch requests", prepare_batches.main),
    ]

    if not args.skip_batch:
        steps.extend([
            ("Step 3: Submit batch", submit_batch.main),
            ("Step 4: Collect results", collect_results.main),
        ])
    else:
        print("\n--- Skipping batch submit/collect (--skip-batch) ---")

    steps.extend([
        ("Step 5: Parse labels", parse_labels.main),
        ("Step 6: Validate against ground truth", validate_labels.main),
    ])

    for title, func in steps:
        print(f"\n{'#' * 60}")
        print(f"# {title}")
        print(f"{'#' * 60}")
        try:
            func()
        except SystemExit as e:
            if e.code and e.code != 0:
                print(f"\n{title} failed with exit code {e.code}")
                sys.exit(e.code)
        except Exception as e:
            print(f"\n{title} failed: {e}")
            raise

    print(f"\n{'#' * 60}")
    print("# Pipeline complete!")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
