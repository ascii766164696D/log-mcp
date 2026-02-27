from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..classifier_bridge import get_classifier
from ..util import validate_file


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

        result = clf.classify_file(file_path, threshold, max_lines, max_look_lines)

        total = result["total_lines"]
        look = result["look_count"]
        skip = result["skip_count"]
        look_lines = result["look_lines"]
        time_s = result["processing_time_s"]
        rate = result["lines_per_second"]

        look_pct = (look / total * 100) if total > 0 else 0.0

        parts: list[str] = []

        if output == "summary":
            parts.append(f"File: {file_path}")
            parts.append(f"Lines: {total:,} total | {look:,} LOOK ({look_pct:.1f}%) | {skip:,} SKIP")
            parts.append(f"Performance: {time_s:.2f}s ({rate:,.0f} lines/sec)")
            parts.append(f"Threshold: {threshold}")

            if look_lines:
                # Confidence distribution
                probs = [p for _, p, _ in look_lines]
                high = sum(1 for p in probs if p >= 0.9)
                med = sum(1 for p in probs if 0.7 <= p < 0.9)
                low = sum(1 for p in probs if p < 0.7)
                parts.append(f"Confidence: {high} high (>=0.9) | {med} medium (0.7-0.9) | {low} low (<0.7)")

                # Sample LOOK lines (up to 30 for summary)
                sample_count = min(30, len(look_lines))
                parts.append("")
                parts.append(f"--- Sample LOOK lines ({sample_count} of {len(look_lines)} captured, {look:,} total) ---")
                for line_no, prob, text in look_lines[:sample_count]:
                    parts.append(f"L{line_no} [{prob:.3f}] {text}")

                if len(look_lines) > sample_count:
                    parts.append(f"... ({len(look_lines) - sample_count} more captured lines)")

        elif output == "look_only":
            parts.append(f"File: {file_path}")
            parts.append(f"Lines: {total:,} total | {look:,} LOOK ({look_pct:.1f}%)")
            parts.append(f"Showing {len(look_lines)} of {look:,} LOOK lines (threshold={threshold})")
            parts.append("")
            for line_no, prob, text in look_lines:
                parts.append(f"L{line_no} [{prob:.3f}] {text}")

            if look > len(look_lines):
                parts.append(f"... ({look - len(look_lines)} more LOOK lines not shown, increase max_look_lines)")

        else:
            return f"Error: Unknown output format '{output}'. Use 'summary' or 'look_only'."

        return "\n".join(parts)
