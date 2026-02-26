from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..formatting import format_entry
from ..parsing.base import LogEntry, count_lines
from ..parsing.detector import detect_parser
from ..util import entry_matches_time_range, parse_datetime, validate_file


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def get_log_segment(
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        max_lines: int = 200,
    ) -> str:
        """Extract a segment of a log file by line range or time range.

        Use line ranges for precise extraction (e.g., around a known error line).
        Use time ranges to get all entries within a time window.
        """
        err = validate_file(file_path)
        if err:
            return f"Error: {err}"

        parser = detect_parser(file_path)
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)

        use_time = start_dt is not None or end_dt is not None
        sl = start_line or 1
        el = end_line

        entries: list[LogEntry] = []
        truncated = False

        for entry in parser.parse_file(file_path, start_line=sl, end_line=el):
            if use_time:
                in_range = entry_matches_time_range(entry, start_dt, end_dt)
                if in_range is False:
                    # If we already collected entries and went past end_time, stop
                    if end_dt and entry.timestamp and entry.timestamp > end_dt:
                        break
                    continue

            entries.append(entry)
            if len(entries) >= max_lines:
                truncated = True
                break

        total = count_lines(file_path)

        # Build plain text output
        header = f"Segment: {len(entries)} entries ({total} total)"
        if truncated:
            header += " [truncated]"

        parts: list[str] = [header]
        if entries:
            parts.append("")
            for entry in entries:
                parts.append(format_entry(entry))

        return "\n".join(parts)
