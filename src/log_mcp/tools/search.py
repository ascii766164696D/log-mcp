from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

from ..formatting import format_entry
from ..parsing.base import LogEntry
from ..parsing.detector import detect_parser
from ..util import (
    entry_matches_time_range,
    normalize_level,
    parse_datetime,
    validate_file,
)


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def search_logs(
        file_path: str,
        pattern: str | None = None,
        log_level: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        max_results: int = 50,
    ) -> str:
        """Search log entries by regex pattern, log level, and/or time range.

        Returns matching entries (up to max_results). Combine filters to narrow results.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        regex = None
        if pattern:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

        target_level = normalize_level(log_level)
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)

        parser = detect_parser(file_path)
        matches: list[LogEntry] = []
        scanned = 0

        for entry in parser.parse_file(file_path):
            scanned += 1

            # Level filter
            if target_level and normalize_level(entry.level) != target_level:
                continue

            # Time filter
            if start_dt or end_dt:
                in_range = entry_matches_time_range(entry, start_dt, end_dt)
                if in_range is False:
                    continue

            # Pattern filter
            if regex and not regex.search(entry.raw):
                continue

            matches.append(entry)
            if len(matches) >= max_results:
                break

        # Build plain text output
        truncated = len(matches) >= max_results
        header = f"Matches: {len(matches)} (scanned {scanned})"
        if truncated:
            header += " [truncated]"

        parts: list[str] = [header]
        if matches:
            parts.append("")
            for entry in matches:
                parts.append(format_entry(entry))

        return "\n".join(parts)
