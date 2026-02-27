from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

from ..classifier_bridge import classify_and_extract
from ..formatting import format_entry
from ..parsing.base import LogEntry
from ..parsing.detector import detect_parser
from ..util import (
    entry_matches_time_range,
    normalize_level,
    parse_datetime,
    validate_file,
)

_ERROR_SEARCH_RE = re.compile(
    r"error|fatal|fail|exception|crash|timeout|panic|denied", re.IGNORECASE
)

_CLASSIFIER_LEVELS = {"ERROR", "FATAL", "CRITICAL", "WARN", "WARNING"}


def _should_use_classifier(pattern: str | None, log_level: str | None) -> bool:
    """Decide whether the classifier would help this search."""
    if log_level and log_level.upper() in _CLASSIFIER_LEVELS:
        return True
    if pattern and _ERROR_SEARCH_RE.search(pattern):
        return True
    return False


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

        # Fast path: use Rust classifier when searching for errors/anomalies
        if _should_use_classifier(pattern, log_level):
            result = classify_and_extract(file_path, threshold=0.3)
            if result is not None:
                classified, total_lines = result
                matches: list[LogEntry] = []

                for cl in classified:
                    # Level filter
                    if target_level and normalize_level(cl.level) != target_level:
                        continue

                    # Time filter
                    if start_dt or end_dt:
                        if cl.timestamp is None:
                            pass  # no timestamp — don't filter out
                        else:
                            ts = cl.timestamp
                            if start_dt and start_dt.tzinfo is None and ts.tzinfo is not None:
                                ts = ts.replace(tzinfo=None)
                            if end_dt and end_dt.tzinfo is None and ts.tzinfo is not None:
                                ts = ts.replace(tzinfo=None)
                            if start_dt and ts < start_dt:
                                continue
                            if end_dt and ts > end_dt:
                                continue

                    # Pattern filter
                    if regex and not regex.search(cl.text):
                        continue

                    matches.append(LogEntry(
                        line_number=cl.line_number,
                        raw=cl.text,
                        timestamp=cl.timestamp,
                        level=cl.level,
                        message=cl.message,
                    ))
                    if len(matches) >= max_results:
                        break

                truncated = len(matches) >= max_results
                header = f"Matches: {len(matches)} (scanned {total_lines})"
                if truncated:
                    header += " [truncated]"

                parts: list[str] = [header]
                if matches:
                    parts.append("")
                    for entry in matches:
                        parts.append(format_entry(entry))

                return "\n".join(parts)

        # Slow path: full Python parsing
        parser = detect_parser(file_path)
        matches = []
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

        parts = [header]
        if matches:
            parts.append("")
            for entry in matches:
                parts.append(format_entry(entry))

        return "\n".join(parts)
