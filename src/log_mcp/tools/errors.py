from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

from ..normalize import normalize
from ..parsing.detector import detect_parser
from ..util import validate_file

_ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}

_MAX_STACK_TRACE_LINES = 10

_CONTENT_ERROR_RE = re.compile(
    r"##\[error\]"
    r"|(?:^|\s)error[:\s]"
    r"|(?:^|\s)fatal[:\s]"
    r"|\bfailed\b"
    r"|\bPermission denied\b"
    r"|\bcommand not found\b"
    r"|\bNo such file or directory\b"
    r"|\bunable to\b"
    r"|\bExited with code [1-9]"
    r"|\bsegmentation fault\b"
    r"|\bcannot find\b"
    r"|\bCould not\b"
    r"|\bnot found\b",
    re.IGNORECASE,
)


def _group_errors(
    entries: list[tuple],
    include_stack_traces: bool,
    max_unique_errors: int,
) -> tuple[int, dict[str, dict]]:
    """Group error entries by normalized fingerprint.

    Args:
        entries: list of (entry, fingerprint) tuples
        include_stack_traces: whether to capture stack traces
        max_unique_errors: max number of unique groups to track

    Returns:
        (total_errors, groups_dict) where groups_dict maps fingerprint -> group info
    """
    groups: dict[str, dict] = {}
    total_errors = 0

    for entry, fp in entries:
        total_errors += 1

        if fp not in groups:
            if len(groups) >= max_unique_errors:
                continue
            first_seen = {
                "line": entry.line_number,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            }
            groups[fp] = {
                "fingerprint": fp,
                "count": 0,
                "first_seen": first_seen,
                "last_seen": dict(first_seen),
                "stack_trace": None,
            }
            if include_stack_traces and "\n" in entry.raw:
                lines = entry.raw.split("\n")
                if len(lines) > 1:
                    trace_lines = lines[1:]
                    if len(trace_lines) > _MAX_STACK_TRACE_LINES:
                        trace_lines = trace_lines[:_MAX_STACK_TRACE_LINES]
                        trace_lines.append(
                            f"... ({len(lines) - 1 - _MAX_STACK_TRACE_LINES} more lines)"
                        )
                    groups[fp]["stack_trace"] = "\n".join(trace_lines)

        groups[fp]["count"] += 1
        groups[fp]["last_seen"] = {
            "line": entry.line_number,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        }

    return total_errors, groups


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def analyze_errors(
        file_path: str,
        include_stack_traces: bool = True,
        max_unique_errors: int = 30,
    ) -> str:
        """Analyze error entries: deduplicate by fingerprint, count frequencies, extract stack traces.

        Groups similar error messages together even when they differ in numbers, IDs, or timestamps.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        parser = detect_parser(file_path)

        # Pass 1: filter by log level
        level_entries = []
        all_entries = []
        for entry in parser.parse_file(file_path):
            all_entries.append(entry)
            if entry.level in _ERROR_LEVELS:
                level_entries.append((entry, normalize(entry.message)))

        used_heuristics = False

        if level_entries:
            total_errors, groups = _group_errors(
                level_entries, include_stack_traces, max_unique_errors
            )
        else:
            # Pass 2 fallback: content-based heuristics
            content_entries = []
            for entry in all_entries:
                if _CONTENT_ERROR_RE.search(entry.message):
                    content_entries.append((entry, normalize(entry.message)))

            total_errors, groups = _group_errors(
                content_entries, include_stack_traces, max_unique_errors
            )
            used_heuristics = True

        # Sort by frequency descending
        sorted_groups = sorted(groups.values(), key=lambda g: g["count"], reverse=True)

        # Build summary
        summary_parts = [f"{total_errors} errors in {len(groups)} groups."]
        if used_heuristics:
            summary_parts.append(
                "(No standard log levels found; used content heuristics.)"
            )
        if sorted_groups:
            top_items = []
            for g in sorted_groups[:3]:
                top_items.append(f"'{g['fingerprint']}' ({g['count']}x)")
            summary_parts.append("Top: " + ", ".join(top_items) + ".")

        # Build plain text output
        parts: list[str] = []
        parts.append(f"Summary: {' '.join(summary_parts)}")

        for g in sorted_groups:
            parts.append("")
            parts.append(f"--- {g['count']}x ---")
            parts.append(f"Fingerprint: {g['fingerprint']}")

            first = g["first_seen"]
            last = g["last_seen"]
            first_str = f"L{first['line']}"
            if first["timestamp"]:
                first_str += f" {first['timestamp']}"
            last_str = f"L{last['line']}"
            if last["timestamp"]:
                last_str += f" {last['timestamp']}"
            parts.append(f"First: {first_str}")
            parts.append(f"Last:  {last_str}")

            if g["stack_trace"]:
                parts.append("Stack trace:")
                for trace_line in g["stack_trace"].split("\n"):
                    parts.append(f"  {trace_line}")

        return "\n".join(parts)
