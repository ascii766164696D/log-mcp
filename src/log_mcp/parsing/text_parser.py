from __future__ import annotations

import re
from datetime import datetime

from .base import LogEntry, LogParser

# Patterns for common text log formats.
# Each pattern must have named groups: timestamp, and optionally level, message.
_PATTERNS = [
    # 2024-01-15 10:30:45.123 ERROR some message
    # 2024-01-15T10:30:45.123Z ERROR some message
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
        r"\s+(?P<level>[A-Z]+)\s+"
        r"(?P<message>.*)$"
    ),
    # [2024-01-15 10:30:45] [ERROR] some message
    re.compile(
        r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]"
        r"\s+\[?(?P<level>[A-Z]+)\]?\s+"
        r"(?P<message>.*)$"
    ),
    # ERROR 2024-01-15 10:30:45 some message
    re.compile(
        r"^(?P<level>[A-Z]+)\s+"
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
        r"\s+(?P<message>.*)$"
    ),
    # Syslog: Jan 15 10:30:45 hostname process[pid]: message
    re.compile(
        r"^(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
        r"\s+\S+\s+\S+:\s+"
        r"(?P<message>.*)$"
    ),
    # Timestamp only, no level
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
        r"\s+(?P<message>.*)$"
    ),
]

_TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%b %d %H:%M:%S",
    "%y/%m/%d %H:%M:%S",
]

# Levels that might appear in log messages
_KNOWN_LEVELS = {"TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"}


def _parse_timestamp(value: str) -> datetime | None:
    # Normalize comma decimal separator to dot
    value = value.replace(",", ".")
    # Truncate fractional seconds to at most 6 digits (Python limit)
    value = re.sub(r"(\.\d{6})\d+", r"\1", value)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class TextLogParser(LogParser):
    """Regex-based parser for common text log formats.

    On first successful parse, locks to that pattern for the rest of the file.
    """

    def __init__(self) -> None:
        self._locked_pattern: re.Pattern | None = None

    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        if not line.strip():
            return None

        if self._locked_pattern is not None:
            return self._try_pattern(self._locked_pattern, line, line_number)

        for pattern in _PATTERNS:
            entry = self._try_pattern(pattern, line, line_number)
            if entry is not None:
                self._locked_pattern = pattern
                return entry

        # No pattern matched — treat as a continuation line
        return None

    def _try_pattern(
        self, pattern: re.Pattern, line: str, line_number: int
    ) -> LogEntry | None:
        m = pattern.match(line)
        if m is None:
            return None

        groups = m.groupdict()
        ts_str = groups.get("timestamp", "")
        timestamp = _parse_timestamp(ts_str) if ts_str else None
        level = groups.get("level")
        if level and level.upper() not in _KNOWN_LEVELS:
            level = None
        elif level:
            level = level.upper()
        message = groups.get("message", line)

        return LogEntry(
            line_number=line_number,
            raw=line,
            timestamp=timestamp,
            level=level,
            message=message,
        )
