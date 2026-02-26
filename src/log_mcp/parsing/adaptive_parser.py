from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from .base import LogEntry, LogParser
from .text_parser import _KNOWN_LEVELS, _parse_timestamp

# Broad regex that finds an ISO-ish timestamp anywhere in a line.
# Captures: 2024-01-15T10:30:45.1234567Z, 2024-01-15 10:30:45+00:00, etc.
# Also matches: 17/06/08 13:33:49 (2-digit year with slashes, common in Spark/Log4j)
_TS_ANYWHERE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"|(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
)


def _extract_ts(m: re.Match) -> str:
    """Return the matched timestamp string from either alternative."""
    return m.group(1) or m.group(2)

# Delimiters to try, in priority order.
_DELIMITERS = ["\t", "|"]


class AdaptiveTextParser(LogParser):
    """Parser that learns log line structure from a sample of lines."""

    def __init__(
        self,
        *,
        delimiter: str | None,
        ts_col: int,
        level_after_ts: bool,
        prefix_col_count: int,
        ts_offset: int | None = None,
    ) -> None:
        self._delimiter = delimiter
        self._ts_col = ts_col
        self._level_after_ts = level_after_ts
        self._prefix_col_count = prefix_col_count
        self._ts_offset = ts_offset

    @classmethod
    def from_sample(cls, lines: list[str]) -> AdaptiveTextParser | None:
        """Analyze sample lines, return parser if a consistent format is found."""
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return None

        # Try delimiter-based detection first
        result = cls._try_delimiters(non_empty)
        if result is not None:
            return result

        # Try fixed-offset detection
        return cls._try_fixed_offset(non_empty)

    @classmethod
    def _try_delimiters(cls, lines: list[str]) -> AdaptiveTextParser | None:
        for delim in _DELIMITERS:
            result = cls._try_delimiter(lines, delim)
            if result is not None:
                return result
        return None

    @classmethod
    def _try_delimiter(
        cls, lines: list[str], delim: str
    ) -> AdaptiveTextParser | None:
        ts_cols: list[int] = []
        col_counts: list[int] = []

        for line in lines:
            parts = line.split(delim)
            if len(parts) < 2:
                continue
            col_counts.append(len(parts))
            # Find which column contains a timestamp
            for i, part in enumerate(parts):
                part = part.strip()
                # Strip leading BOM if present
                if part.startswith("\ufeff"):
                    part = part[1:]
                m = _TS_ANYWHERE.search(part)
                if m and _parse_timestamp(_extract_ts(m)) is not None:
                    ts_cols.append(i)
                    break

        if not ts_cols:
            return None

        # Need at least 50% of lines to have a timestamp in the same column
        col_counter = Counter(ts_cols)
        best_col, best_count = col_counter.most_common(1)[0]
        if best_count / len(lines) < 0.5:
            return None

        # Determine prefix column count (columns before the timestamp column)
        prefix_col_count = best_col

        # Check if a known level typically follows the timestamp
        level_after_ts = cls._detect_level_after_ts(lines, delim, best_col)

        return cls(
            delimiter=delim,
            ts_col=best_col,
            level_after_ts=level_after_ts,
            prefix_col_count=prefix_col_count,
        )

    @classmethod
    def _try_fixed_offset(cls, lines: list[str]) -> AdaptiveTextParser | None:
        offsets: list[int] = []

        for line in lines:
            m = _TS_ANYWHERE.search(line)
            if m and _parse_timestamp(_extract_ts(m)) is not None:
                offsets.append(m.start())

        if not offsets:
            return None

        offset_counter = Counter(offsets)
        best_offset, best_count = offset_counter.most_common(1)[0]
        if best_count / len(lines) < 0.5:
            return None

        level_after_ts = cls._detect_level_after_ts_at_offset(lines, best_offset)

        return cls(
            delimiter=None,
            ts_col=0,
            level_after_ts=level_after_ts,
            prefix_col_count=0,
            ts_offset=best_offset,
        )

    @classmethod
    def _detect_level_after_ts(
        cls, lines: list[str], delim: str, ts_col: int
    ) -> bool:
        level_count = 0
        checked = 0
        for line in lines:
            parts = line.split(delim)
            if ts_col >= len(parts):
                continue
            # Look at text after the timestamp column
            rest_parts = parts[ts_col + 1 :]
            if not rest_parts:
                continue
            rest = (delim.join(rest_parts)).strip()
            checked += 1
            first_word = rest.split()[0] if rest.split() else ""
            if first_word.upper() in _KNOWN_LEVELS:
                level_count += 1
        return checked > 0 and level_count / checked >= 0.3

    @classmethod
    def _detect_level_after_ts_at_offset(
        cls, lines: list[str], offset: int
    ) -> bool:
        level_count = 0
        checked = 0
        for line in lines:
            m = _TS_ANYWHERE.search(line)
            if not m or m.start() != offset:
                continue
            rest = line[m.end() :].strip()
            checked += 1
            first_word = rest.split()[0] if rest.split() else ""
            if first_word.upper() in _KNOWN_LEVELS:
                level_count += 1
        return checked > 0 and level_count / checked >= 0.3

    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        if not line.strip():
            return None

        if self._delimiter is not None:
            return self._parse_delimited(line, line_number)
        return self._parse_fixed_offset(line, line_number)

    def _parse_delimited(self, line: str, line_number: int) -> LogEntry | None:
        parts = line.split(self._delimiter)
        if self._ts_col >= len(parts):
            return None

        ts_part = parts[self._ts_col].strip()
        # Strip BOM
        if ts_part.startswith("\ufeff"):
            ts_part = ts_part[1:]

        m = _TS_ANYWHERE.search(ts_part)
        if not m:
            return None

        timestamp = _parse_timestamp(_extract_ts(m))

        # Collect prefix columns as extra metadata
        extra: dict = {}
        prefix_parts = parts[: self._ts_col]
        if prefix_parts:
            extra["prefix"] = self._delimiter.join(prefix_parts)

        # Everything after the timestamp column is the message
        rest_parts = parts[self._ts_col + 1 :]
        rest = (self._delimiter.join(rest_parts)).strip() if rest_parts else ""

        # The timestamp column may have text after the timestamp itself
        after_ts_in_col = ts_part[m.end() :].strip()
        if after_ts_in_col:
            rest = after_ts_in_col + (" " + rest if rest else "")

        level: str | None = None
        message = rest

        if self._level_after_ts and rest:
            words = rest.split(None, 1)
            if words and words[0].upper() in _KNOWN_LEVELS:
                level = words[0].upper()
                message = words[1] if len(words) > 1 else ""

        return LogEntry(
            line_number=line_number,
            raw=line,
            timestamp=timestamp,
            level=level,
            message=message,
            extra=extra,
        )

    def _parse_fixed_offset(self, line: str, line_number: int) -> LogEntry | None:
        m = _TS_ANYWHERE.search(line)
        if not m:
            return None

        timestamp = _parse_timestamp(_extract_ts(m))
        prefix = line[: m.start()].strip()
        rest = line[m.end() :].strip()

        extra: dict = {}
        if prefix:
            extra["prefix"] = prefix

        level: str | None = None
        message = rest

        if self._level_after_ts and rest:
            words = rest.split(None, 1)
            if words and words[0].upper() in _KNOWN_LEVELS:
                level = words[0].upper()
                message = words[1] if len(words) > 1 else ""

        return LogEntry(
            line_number=line_number,
            raw=line,
            timestamp=timestamp,
            level=level,
            message=message,
            extra=extra,
        )
