from __future__ import annotations

import gzip
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import IO, Iterator


@dataclass
class LogEntry:
    line_number: int
    raw: str
    timestamp: datetime | None = None
    level: str | None = None
    message: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"line": self.line_number, "message": self.message}
        if self.timestamp:
            d["timestamp"] = self.timestamp.isoformat()
        if self.level:
            d["level"] = self.level
        if self.extra:
            d["extra"] = self.extra
        return d


class LogParser(ABC):
    """Base class for log parsers."""

    @abstractmethod
    def parse_line(self, line: str, line_number: int) -> LogEntry | None:
        """Parse a single line. Returns None if the line is a continuation."""
        ...

    def parse_file(
        self,
        file_path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_entries: int | None = None,
    ) -> Iterator[LogEntry]:
        """Parse a file yielding LogEntry objects.

        Handles multi-line entries by appending continuation lines to the
        previous entry.
        """
        current: LogEntry | None = None
        count = 0

        for line_number, line in _iter_lines(file_path):
            if end_line is not None and line_number > end_line:
                break
            if line_number < start_line:
                continue

            entry = self.parse_line(line, line_number)
            if entry is not None:
                if current is not None:
                    yield current
                    count += 1
                    if max_entries is not None and count >= max_entries:
                        return
                current = entry
            elif current is not None:
                # Continuation line — append to previous entry
                current.raw += "\n" + line
                current.message += "\n" + line

        if current is not None:
            yield current


def _iter_lines(file_path: str) -> Iterator[tuple[int, str]]:
    """Yield (1-based line_number, stripped_line) from a file.

    Transparently handles .gz files.
    """
    opener: type[IO] | type[gzip.GzipFile]
    if file_path.endswith(".gz"):
        fh = gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    else:
        fh = open(file_path, encoding="utf-8", errors="replace")

    with fh:
        for i, line in enumerate(fh, start=1):
            yield i, line.rstrip("\n\r")


def count_lines(file_path: str) -> int:
    """Efficiently count lines without loading the whole file."""
    count = 0
    if file_path.endswith(".gz"):
        fh = gzip.open(file_path, "rb")
    else:
        fh = open(file_path, "rb")
    with fh:
        for _ in fh:
            count += 1
    return count


def file_size(file_path: str) -> int:
    return os.path.getsize(file_path)
