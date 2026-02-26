from __future__ import annotations

import os
from datetime import datetime

from .parsing.base import LogEntry


def validate_file(file_path: str) -> str | None:
    """Return an error message if the file is not readable, else None."""
    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"
    if not os.access(file_path, os.R_OK):
        return f"File not readable: {file_path}"
    return None


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def parse_datetime(value: str | None) -> datetime | None:
    """Best-effort parse of a user-supplied datetime string."""
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def normalize_level(level: str | None) -> str | None:
    """Normalize log level to uppercase canonical form."""
    if not level:
        return None
    level = level.upper().strip()
    if level == "WARNING":
        return "WARN"
    return level


def entry_matches_time_range(
    entry: LogEntry,
    start_time: datetime | None,
    end_time: datetime | None,
) -> bool | None:
    """Check if an entry falls within a time range.

    Returns None if the entry has no timestamp (skip filtering).
    """
    if entry.timestamp is None:
        return None
    ts = entry.timestamp
    # Strip tzinfo for naive comparison if needed
    if start_time and start_time.tzinfo is None and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    if end_time and end_time.tzinfo is None and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    if start_time and ts < start_time:
        return False
    if end_time and ts > end_time:
        return False
    return True
