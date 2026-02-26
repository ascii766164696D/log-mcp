from __future__ import annotations

from .parsing.base import LogEntry


def format_entry(entry: LogEntry) -> str:
    """Format a log entry as 'L{line} [{timestamp}] {level} {message}'.

    Multi-line entries indent continuation lines with spaces.
    """
    parts = [f"L{entry.line_number}"]
    if entry.timestamp:
        parts.append(f"[{entry.timestamp.isoformat()}]")
    if entry.level:
        parts.append(entry.level)
    prefix = " ".join(parts)

    lines = entry.message.split("\n")
    if len(lines) <= 1:
        return f"{prefix} {entry.message}"

    indent = " " * (len(f"L{entry.line_number}") + 1)
    result = f"{prefix} {lines[0]}"
    for line in lines[1:]:
        result += f"\n{indent}{line}"
    return result
