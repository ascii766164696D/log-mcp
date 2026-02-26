from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..parsing.base import LogEntry, count_lines, file_size, _iter_lines
from ..parsing.detector import detect_parser
from ..util import format_size, validate_file


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def log_overview(file_path: str, sample_lines: int = 10) -> str:
        """Quick scan of a log file: size, line count, time range, level distribution, and head/tail samples.

        Use this as the first step when investigating a log file.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        size = file_size(file_path)
        total_lines = count_lines(file_path)

        parser = detect_parser(file_path)

        # Collect head and tail raw lines for samples
        head_lines: list[str] = []
        tail_lines: list[str] = []
        for ln, line in _iter_lines(file_path):
            if ln <= sample_lines:
                head_lines.append(line)
            tail_lines.append(line)
            if len(tail_lines) > sample_lines:
                tail_lines.pop(0)

        # Parse entries for stats
        level_counts: dict[str, int] = {}
        first_ts = None
        last_ts = None

        for entry in parser.parse_file(file_path):
            if entry.level:
                level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            if entry.timestamp:
                if first_ts is None:
                    first_ts = entry.timestamp
                last_ts = entry.timestamp

        # Build plain text output
        parts: list[str] = []
        parts.append(f"File: {file_path}")
        parts.append(
            f"Size: {format_size(size)} ({size} bytes) | Lines: {total_lines}"
        )

        if first_ts or last_ts:
            start = first_ts.isoformat() if first_ts else "?"
            end = last_ts.isoformat() if last_ts else "?"
            parts.append(f"Time: {start} \u2192 {end}")

        if level_counts:
            levels_str = " ".join(
                f"{k}={v}" for k, v in level_counts.items()
            )
            parts.append(f"Levels: {levels_str}")

        if head_lines:
            parts.append("")
            parts.append("--- Head ---")
            parts.extend(head_lines)

        if tail_lines:
            parts.append("")
            parts.append("--- Tail ---")
            parts.extend(tail_lines)

        return "\n".join(parts)
