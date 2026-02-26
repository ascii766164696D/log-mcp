from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP

from ..normalize import normalize_extended
from ..parsing.detector import detect_parser
from ..util import validate_file

_BUCKET_SECONDS = {
    "1s": 1,
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def _choose_bucket_size(start: datetime, end: datetime) -> str:
    """Choose a bucket size that produces roughly 20-60 buckets."""
    span = (end - start).total_seconds()
    for label, secs in _BUCKET_SECONDS.items():
        buckets = span / secs
        if 10 <= buckets <= 80:
            return label
    return "1h"


def _bucket_key(ts: datetime, bucket_seconds: int) -> str:
    """Round a timestamp down to its bucket."""
    epoch = ts.timestamp()
    bucketed = int(epoch // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(bucketed).strftime("%Y-%m-%d %H:%M:%S")


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def log_stats(
        file_path: str,
        bucket_size: str = "auto",
        top_patterns: int = 20,
    ) -> str:
        """Compute log statistics: volume histogram over time, level breakdown, and top repeated message patterns.

        Useful for spotting traffic spikes, error bursts, or noisy log sources.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        parser = detect_parser(file_path)

        timestamps: list[datetime] = []
        level_counts: dict[str, int] = {}
        pattern_counter: Counter[str] = Counter()
        total_entries = 0

        for entry in parser.parse_file(file_path):
            total_entries += 1
            if entry.timestamp:
                timestamps.append(entry.timestamp)
            if entry.level:
                level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            pattern_counter[normalize_extended(entry.message)] += 1

        # Build histogram
        histogram: dict[str, int] = {}
        chosen_bucket = bucket_size
        if timestamps:
            first_ts = min(timestamps)
            last_ts = max(timestamps)

            if chosen_bucket == "auto":
                chosen_bucket = _choose_bucket_size(first_ts, last_ts)

            bs = _BUCKET_SECONDS.get(chosen_bucket, 3600)

            for ts in timestamps:
                key = _bucket_key(ts, bs)
                histogram[key] = histogram.get(key, 0) + 1

        # Top patterns
        top = pattern_counter.most_common(top_patterns)

        # Build summary
        summary_parts = [f"{total_entries} entries"]
        if timestamps:
            span = (last_ts - first_ts).total_seconds()
            if span >= 3600:
                summary_parts[0] += f" over {span / 3600:.1f}h"
            elif span >= 60:
                summary_parts[0] += f" over {span / 60:.0f}m"
            else:
                summary_parts[0] += f" over {span:.0f}s"
        summary_parts[0] += "."

        error_count = sum(
            v for k, v in level_counts.items()
            if k in {"ERROR", "FATAL", "CRITICAL"}
        )
        if total_entries > 0 and error_count > 0:
            pct = error_count / total_entries * 100
            summary_parts.append(f"{pct:.1f}% errors.")

        if histogram:
            peak_key = max(histogram, key=histogram.get)  # type: ignore[arg-type]
            summary_parts.append(
                f"Peak: {histogram[peak_key]} at {peak_key}."
            )

        # Build plain text output
        parts: list[str] = []
        parts.append(f"Summary: {' '.join(summary_parts)}")

        if level_counts:
            levels_str = " ".join(f"{k}={v}" for k, v in level_counts.items())
            parts.append("")
            parts.append(f"Levels: {levels_str}")

        if histogram:
            parts.append("")
            parts.append(f"Histogram ({chosen_bucket} buckets):")
            for bucket_time, count in histogram.items():
                parts.append(f"  {bucket_time}  {count}")

        if top:
            parts.append("")
            parts.append("Top patterns:")
            for pattern, count in top:
                parts.append(f"  {count:>4}x {pattern}")

        return "\n".join(parts)
