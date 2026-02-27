"""Shared bridge to the Rust LOOK/SKIP classifier.

Provides a single get_classifier() used by all tools, plus helpers for
running the classifier as a pre-filter (e.g., for analyze_errors and
search_logs).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

_classifier = None
_transformer = None

# Lightweight level extraction — much cheaper than full log parsing.
_LEVEL_RE = re.compile(
    r"\b(EMERGENCY|FATAL|CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|NOTICE)\b",
    re.IGNORECASE,
)

# ISO-ish timestamp extraction (covers most common formats).
_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)

_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


@dataclass
class ClassifiedLine:
    line_number: int
    probability: float
    text: str
    level: str | None
    timestamp: datetime | None
    message: str


def get_classifier():
    """Lazy-load the Rust classifier on first call. Returns None if unavailable."""
    global _classifier
    if _classifier is not None:
        return _classifier

    try:
        from look_skip_classifier import LookSkipClassifier
    except ImportError:
        return None

    model_path = os.environ.get("LOOK_SKIP_MODEL_PATH")
    if not model_path:
        here = os.path.dirname(__file__)
        candidate = os.path.normpath(
            os.path.join(here, "..", "..", "data", "models", "look_skip_model.json")
        )
        if os.path.isfile(candidate):
            model_path = candidate

    if not model_path or not os.path.isfile(model_path):
        return None

    _classifier = LookSkipClassifier(model_path)
    return _classifier


def get_transformer():
    """Lazy-load the Rust transformer classifier. Returns None if unavailable."""
    global _transformer
    if _transformer is not None:
        return _transformer

    try:
        from look_skip_classifier import TransformerClassifier
    except ImportError:
        return None

    model_dir = os.environ.get("LOOK_SKIP_TRANSFORMER_PATH")
    if not model_dir:
        here = os.path.dirname(__file__)
        candidate = os.path.normpath(
            os.path.join(here, "..", "..", "data", "models", "transformer", "export")
        )
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "model.safetensors")
        ):
            model_dir = candidate

    if not model_dir or not os.path.isdir(model_dir):
        return None

    _transformer = TransformerClassifier(model_dir)
    return _transformer


def _parse_timestamp(text: str) -> datetime | None:
    m = _TIMESTAMP_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def extract_fields(line_no: int, prob: float, text: str) -> ClassifiedLine:
    """Extract level and timestamp from raw text using lightweight regexes."""
    level_match = _LEVEL_RE.search(text)
    level = level_match.group(1).upper() if level_match else None
    if level == "WARNING":
        level = "WARN"
    timestamp = _parse_timestamp(text)
    return ClassifiedLine(
        line_number=line_no,
        probability=prob,
        text=text,
        level=level,
        timestamp=timestamp,
        message=text,
    )


def classify_and_extract(
    file_path: str,
    threshold: float = 0.3,
    max_look_lines: int = 999_999,
) -> tuple[list[ClassifiedLine], int] | None:
    """Run classifier on file, return (classified_lines, total_lines) or None if unavailable."""
    clf = get_classifier()
    if clf is None:
        return None

    result = clf.classify_file(file_path, threshold, 0, max_look_lines)
    total_lines = result["total_lines"]
    look_lines = result["look_lines"]

    classified = [
        extract_fields(line_no, prob, text)
        for line_no, prob, text in look_lines
    ]
    return classified, total_lines
