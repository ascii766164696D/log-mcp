from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..classifier_bridge import get_classifier, get_transformer
from ..util import validate_file

# Batch size for transformer re-scoring.
_TRANSFORMER_BATCH_SIZE = 128


def _rescore_with_transformer(transformer, look_lines, threshold):
    """Re-score TF-IDF LOOK lines with the transformer model.

    Returns (new_look_lines, new_look_count, new_skip_count) where
    new_look_lines uses transformer probabilities and new_skip_count
    is the number of lines the transformer demoted to SKIP.
    """
    rescored_look = []
    demoted = 0

    for i in range(0, len(look_lines), _TRANSFORMER_BATCH_SIZE):
        batch = look_lines[i : i + _TRANSFORMER_BATCH_SIZE]
        texts = [text for _, _, text in batch]
        results = transformer.classify_batch(texts, threshold)

        for (line_no, _tfidf_prob, text), (label, bert_prob) in zip(batch, results):
            if label == "LOOK":
                rescored_look.append((line_no, bert_prob, text))
            else:
                demoted += 1

    return rescored_look, len(rescored_look), demoted


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def classify_lines(
        file_path: str,
        threshold: float = 0.5,
        max_lines: int = 0,
        max_look_lines: int = 200,
        output: str = "summary",
    ) -> str:
        """Classify log lines as LOOK (interesting) or SKIP (routine) using a trained ML model.

        Uses a logistic regression model trained on 17 loghub datasets (345M lines).
        Lines classified as LOOK include errors, warnings, security events, resource
        exhaustion, hardware anomalies, and other operationally significant entries.

        Args:
            file_path: Path to the log file to classify.
            threshold: Probability threshold for LOOK classification (0.0-1.0, default 0.5).
                Lower values capture more lines but with more false positives.
            max_lines: Maximum number of lines to process (0 = all lines).
            max_look_lines: Maximum number of LOOK lines to return in detail (default 200).
            output: Output format - "summary" for overview stats + sample LOOK lines,
                "look_only" for all captured LOOK lines with probabilities.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        clf = get_classifier()
        if clf is None:
            return (
                "Error: LOOK/SKIP classifier not available. "
                "Install look-skip-classifier package and ensure model file exists at "
                "LOOK_SKIP_MODEL_PATH or data/models/look_skip_model.json."
            )

        # Stage 1: TF-IDF classifier scans the full file
        # Use a low threshold to cast a wide net — the transformer refines later.
        transformer = get_transformer()
        tfidf_threshold = threshold * 0.6 if transformer else threshold
        tfidf_max_look = max_look_lines if not transformer else 999_999

        result = clf.classify_file(file_path, tfidf_threshold, max_lines, tfidf_max_look)

        total = result["total_lines"]
        look = result["look_count"]
        skip = result["skip_count"]
        look_lines = result["look_lines"]
        tfidf_time_s = result["processing_time_s"]
        tfidf_rate = result["lines_per_second"]

        # Stage 2: Transformer re-scores TF-IDF LOOK lines
        bert_time_s = 0.0
        tfidf_look = look
        demoted = 0
        if transformer and look_lines:
            import time

            t0 = time.perf_counter()
            look_lines, look, demoted = _rescore_with_transformer(
                transformer, look_lines, threshold
            )
            skip += demoted
            bert_time_s = time.perf_counter() - t0

            # Trim to max_look_lines after re-scoring
            if len(look_lines) > max_look_lines:
                look_lines = look_lines[:max_look_lines]

        total_time_s = tfidf_time_s + bert_time_s
        look_pct = (look / total * 100) if total > 0 else 0.0

        parts: list[str] = []

        if output == "summary":
            parts.append(f"File: {file_path}")
            parts.append(
                f"Lines: {total:,} total | {look:,} LOOK ({look_pct:.1f}%) | {skip:,} SKIP"
            )
            if transformer and bert_time_s > 0:
                parts.append(
                    f"Pipeline: TF-IDF {tfidf_time_s:.2f}s ({tfidf_rate:,.0f} lines/sec, "
                    f"{tfidf_look:,} LOOK) → BERT {bert_time_s:.2f}s "
                    f"({demoted:,} demoted to SKIP)"
                )
                parts.append(f"Total: {total_time_s:.2f}s")
            else:
                parts.append(f"Performance: {tfidf_time_s:.2f}s ({tfidf_rate:,.0f} lines/sec)")
            parts.append(f"Threshold: {threshold}")

            if look_lines:
                # Confidence distribution
                probs = [p for _, p, _ in look_lines]
                high = sum(1 for p in probs if p >= 0.9)
                med = sum(1 for p in probs if 0.7 <= p < 0.9)
                low = sum(1 for p in probs if p < 0.7)
                parts.append(
                    f"Confidence: {high} high (>=0.9) | {med} medium (0.7-0.9) | {low} low (<0.7)"
                )

                # Sample LOOK lines (up to 30 for summary)
                sample_count = min(30, len(look_lines))
                parts.append("")
                parts.append(
                    f"--- Sample LOOK lines ({sample_count} of {len(look_lines)} captured, "
                    f"{look:,} total) ---"
                )
                for line_no, prob, text in look_lines[:sample_count]:
                    parts.append(f"L{line_no} [{prob:.3f}] {text}")

                if len(look_lines) > sample_count:
                    parts.append(f"... ({len(look_lines) - sample_count} more captured lines)")

        elif output == "look_only":
            parts.append(f"File: {file_path}")
            parts.append(f"Lines: {total:,} total | {look:,} LOOK ({look_pct:.1f}%)")
            if transformer and bert_time_s > 0:
                parts.append(
                    f"Pipeline: TF-IDF → BERT ({demoted:,} demoted)"
                )
            parts.append(
                f"Showing {len(look_lines)} of {look:,} LOOK lines (threshold={threshold})"
            )
            parts.append("")
            for line_no, prob, text in look_lines:
                parts.append(f"L{line_no} [{prob:.3f}] {text}")

            if look > len(look_lines):
                parts.append(
                    f"... ({look - len(look_lines)} more LOOK lines not shown, "
                    f"increase max_look_lines)"
                )

        else:
            return f"Error: Unknown output format '{output}'. Use 'summary' or 'look_only'."

        return "\n".join(parts)
