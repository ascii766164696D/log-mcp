from __future__ import annotations

import json
from datetime import datetime

from .base import LogEntry, LogParser

# Common field names for JSON log entries
_TIMESTAMP_KEYS = ("timestamp", "time", "ts", "@timestamp", "datetime", "date")
_LEVEL_KEYS = ("level", "severity", "log_level", "loglevel", "lvl")
_MESSAGE_KEYS = ("message", "msg", "text", "log")


def _find_key(data: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in data:
            return data[k]
    # Case-insensitive fallback
    lower = {k.lower(): v for k, v in data.items()}
    for k in candidates:
        if k in lower:
            return lower[k]
    return None


def _parse_timestamp(value: str | int | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            # Unix epoch seconds (or millis if > 1e12)
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000)
            return datetime.fromtimestamp(value)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        # Try ISO 8601 first
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


class JsonLogParser(LogParser):
    """Parser for JSON-lines log files (one JSON object per line)."""

    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        ts_raw = _find_key(data, _TIMESTAMP_KEYS)
        timestamp = _parse_timestamp(ts_raw)

        level = _find_key(data, _LEVEL_KEYS)
        if isinstance(level, str):
            level = level.upper()

        message = _find_key(data, _MESSAGE_KEYS) or ""

        # Everything else goes into extra
        skip = set()
        for keys in (_TIMESTAMP_KEYS, _LEVEL_KEYS, _MESSAGE_KEYS):
            for k in keys:
                skip.add(k)
                skip.add(k.lower())
        extra = {k: v for k, v in data.items() if k.lower() not in skip}

        return LogEntry(
            line_number=line_number,
            raw=line,
            timestamp=timestamp,
            level=level,
            message=str(message),
            extra=extra,
        )
