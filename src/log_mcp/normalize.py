"""Shared message normalization for log analysis tools."""

from __future__ import annotations

import re

_BASE_SUBS = [
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<HEX>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"\b[a-z]+_(?=[a-z]*\d)[a-z0-9]{4,}\b", re.I), "<ID>"),
    (re.compile(r"\b\d+"), "<N>"),
]

_EXTENDED_SUBS = [
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<HEX>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b[a-z]+_(?=[a-z]*\d)[a-z0-9]{4,}\b", re.I), "<ID>"),
    (re.compile(r"\b\d+"), "<N>"),
    (re.compile(r'"[^"]*"'), '"<STR>"'),
    (re.compile(r"'[^']*'"), "'<STR>'"),
]

_WHITESPACE = re.compile(r"\s+")


def normalize(message: str) -> str:
    """Normalize variable parts (UUIDs, hex, IDs, numbers) for pattern matching."""
    result = message
    for pattern, replacement in _BASE_SUBS:
        result = pattern.sub(replacement, result)
    return _WHITESPACE.sub(" ", result).strip()


def normalize_extended(message: str) -> str:
    """Extended normalization that also replaces IPs and quoted strings."""
    result = message
    for pattern, replacement in _EXTENDED_SUBS:
        result = pattern.sub(replacement, result)
    return _WHITESPACE.sub(" ", result).strip()
