from __future__ import annotations

import json

from .adaptive_parser import AdaptiveTextParser
from .base import LogParser, _iter_lines
from .json_parser import JsonLogParser
from .text_parser import TextLogParser

_SAMPLE_SIZE = 20


def detect_parser(file_path: str) -> LogParser:
    """Auto-detect the log format by sampling the first lines of a file.

    Detection order:
    1. JSON
    2. TextLogParser hardcoded patterns (if ≥50% of sample lines match)
    3. AdaptiveTextParser (timestamp anchor discovery)
    4. Fallback: TextLogParser (raw lines)
    """
    lines: list[str] = []
    for _, line in _iter_lines(file_path):
        lines.append(line)
        if len(lines) >= _SAMPLE_SIZE:
            break

    if not lines:
        return TextLogParser()

    non_empty_lines = [l for l in lines if l.strip()]
    non_empty = len(non_empty_lines)

    # 1. Check if the majority of non-empty lines are valid JSON objects
    json_hits = 0
    for line in non_empty_lines:
        try:
            obj = json.loads(line.strip())
            if isinstance(obj, dict):
                json_hits += 1
        except json.JSONDecodeError:
            pass

    if non_empty > 0 and json_hits / non_empty >= 0.5:
        return JsonLogParser()

    # 2. Probe TextLogParser hardcoded patterns against the sample
    text_parser = TextLogParser()
    text_hits = 0
    for line in non_empty_lines:
        entry = text_parser.parse_line(line, 0)
        if entry is not None:
            text_hits += 1

    if non_empty > 0 and text_hits / non_empty >= 0.5:
        return TextLogParser()

    # 3. Try adaptive format detection
    adaptive = AdaptiveTextParser.from_sample(lines)
    if adaptive is not None:
        return adaptive

    # 4. Fallback
    return TextLogParser()
