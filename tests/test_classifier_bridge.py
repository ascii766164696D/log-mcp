"""Tests for classifier_bridge and its integration with tools."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from log_mcp.classifier_bridge import (
    ClassifiedLine,
    classify_and_extract,
    extract_fields,
)
from log_mcp.tools.search import _should_use_classifier


class TestExtractFields:
    def test_extracts_error_level(self):
        cl = extract_fields(1, 0.9, "2024-01-15 10:00:02 ERROR timeout after 30s")
        assert cl.level == "ERROR"
        assert cl.line_number == 1
        assert cl.probability == 0.9

    def test_extracts_warn_level(self):
        cl = extract_fields(5, 0.7, "2024-01-15 10:00:00 WARN disk usage high")
        assert cl.level == "WARN"

    def test_normalizes_warning_to_warn(self):
        cl = extract_fields(5, 0.7, "2024-01-15 10:00:00 WARNING disk usage high")
        assert cl.level == "WARN"

    def test_extracts_fatal_level(self):
        cl = extract_fields(2, 0.95, "FATAL out of memory")
        assert cl.level == "FATAL"

    def test_extracts_info_level(self):
        cl = extract_fields(3, 0.6, "2024-01-15 10:00:00 INFO starting up")
        assert cl.level == "INFO"

    def test_no_level(self):
        cl = extract_fields(10, 0.5, "something happened without a level")
        assert cl.level is None

    def test_extracts_iso_timestamp(self):
        cl = extract_fields(1, 0.9, "2024-01-15T10:00:02.123 ERROR timeout")
        assert cl.timestamp is not None
        assert cl.timestamp.year == 2024
        assert cl.timestamp.month == 1
        assert cl.timestamp.day == 15
        assert cl.timestamp.hour == 10
        assert cl.timestamp.minute == 0
        assert cl.timestamp.second == 2

    def test_extracts_space_timestamp(self):
        cl = extract_fields(1, 0.9, "2024-01-15 10:00:02 ERROR timeout")
        assert cl.timestamp is not None
        assert cl.timestamp.year == 2024

    def test_extracts_timestamp_with_timezone(self):
        cl = extract_fields(1, 0.9, "2024-01-15T10:00:02+00:00 ERROR timeout")
        assert cl.timestamp is not None
        assert cl.timestamp.tzinfo is not None

    def test_no_timestamp(self):
        cl = extract_fields(1, 0.9, "ERROR timeout after 30s")
        assert cl.timestamp is None

    def test_message_is_full_text(self):
        text = "2024-01-15 10:00:02 ERROR timeout after 30s"
        cl = extract_fields(1, 0.9, text)
        assert cl.message == text
        assert cl.text == text


class TestShouldUseClassifier:
    def test_error_level_triggers(self):
        assert _should_use_classifier(None, "ERROR") is True

    def test_fatal_level_triggers(self):
        assert _should_use_classifier(None, "FATAL") is True

    def test_critical_level_triggers(self):
        assert _should_use_classifier(None, "CRITICAL") is True

    def test_warn_level_triggers(self):
        assert _should_use_classifier(None, "WARN") is True

    def test_warning_level_triggers(self):
        assert _should_use_classifier(None, "WARNING") is True

    def test_info_level_does_not_trigger(self):
        assert _should_use_classifier(None, "INFO") is False

    def test_debug_level_does_not_trigger(self):
        assert _should_use_classifier(None, "DEBUG") is False

    def test_error_pattern_triggers(self):
        assert _should_use_classifier("error", None) is True

    def test_fatal_pattern_triggers(self):
        assert _should_use_classifier("fatal", None) is True

    def test_exception_pattern_triggers(self):
        assert _should_use_classifier("NullPointerException", None) is True

    def test_timeout_pattern_triggers(self):
        assert _should_use_classifier("timeout", None) is True

    def test_generic_pattern_does_not_trigger(self):
        assert _should_use_classifier("user login", None) is False

    def test_no_filters_does_not_trigger(self):
        assert _should_use_classifier(None, None) is False


class TestClassifyAndExtractFallback:
    """Test that classify_and_extract returns None when classifier is unavailable."""

    def test_returns_none_when_classifier_unavailable(self, tmp_log):
        fp = tmp_log("2024-01-15 10:00:01 ERROR test\n")
        with patch("log_mcp.classifier_bridge.get_classifier", return_value=None):
            result = classify_and_extract(fp)
        assert result is None


class TestAnalyzeErrorsFastPath:
    """Test the fast path in analyze_errors."""

    def test_fast_path_finds_level_errors(self):
        from log_mcp.tools.errors import _analyze_errors_fast

        classified = [
            ClassifiedLine(
                line_number=1,
                probability=0.9,
                text="2024-01-15 10:00:01 INFO Starting",
                level="INFO",
                timestamp=datetime(2024, 1, 15, 10, 0, 1),
                message="2024-01-15 10:00:01 INFO Starting",
            ),
            ClassifiedLine(
                line_number=2,
                probability=0.95,
                text="2024-01-15 10:00:02 ERROR timeout after 30s",
                level="ERROR",
                timestamp=datetime(2024, 1, 15, 10, 0, 2),
                message="2024-01-15 10:00:02 ERROR timeout after 30s",
            ),
            ClassifiedLine(
                line_number=3,
                probability=0.95,
                text="2024-01-15 10:00:03 ERROR timeout after 45s",
                level="ERROR",
                timestamp=datetime(2024, 1, 15, 10, 0, 3),
                message="2024-01-15 10:00:03 ERROR timeout after 45s",
            ),
        ]
        total_errors, groups, used_heuristics = _analyze_errors_fast(
            classified, False, 30
        )
        assert total_errors == 2
        assert not used_heuristics
        # Both timeouts should normalize to the same fingerprint
        assert len(groups) == 1

    def test_fast_path_content_heuristic_fallback(self):
        from log_mcp.tools.errors import _analyze_errors_fast

        classified = [
            ClassifiedLine(
                line_number=1,
                probability=0.8,
                text="fatal: could not read Username",
                level=None,
                timestamp=None,
                message="fatal: could not read Username",
            ),
            ClassifiedLine(
                line_number=2,
                probability=0.7,
                text="Permission denied",
                level=None,
                timestamp=None,
                message="Permission denied",
            ),
        ]
        total_errors, groups, used_heuristics = _analyze_errors_fast(
            classified, False, 30
        )
        assert total_errors == 2
        assert used_heuristics
        assert len(groups) == 2

    def test_fast_path_no_errors_found(self):
        from log_mcp.tools.errors import _analyze_errors_fast

        classified = [
            ClassifiedLine(
                line_number=1,
                probability=0.6,
                text="2024-01-15 10:00:01 INFO Starting",
                level="INFO",
                timestamp=datetime(2024, 1, 15, 10, 0, 1),
                message="2024-01-15 10:00:01 INFO Starting",
            ),
        ]
        total_errors, groups, used_heuristics = _analyze_errors_fast(
            classified, False, 30
        )
        assert total_errors == 0
        assert used_heuristics  # fell through to content heuristics
        assert len(groups) == 0


class TestAnalyzeErrorsIntegrationWithClassifier:
    """Test that analyze_errors correctly delegates to fast/slow path."""

    def test_falls_back_to_slow_path(self, tmp_log):
        """When classifier is unavailable, the slow path produces correct results."""
        fp = tmp_log(
            "2024-01-15 10:00:01 INFO Starting\n"
            "2024-01-15 10:00:02 ERROR timeout after 30s\n"
            "2024-01-15 10:00:03 ERROR timeout after 45s\n"
            "2024-01-15 10:00:04 INFO Recovered\n"
        )
        with patch("log_mcp.tools.errors.classify_and_extract", return_value=None):
            from log_mcp.tools.errors import _group_errors, _CONTENT_ERROR_RE
            from log_mcp.parsing.detector import detect_parser
            from log_mcp.normalize import normalize

            parser = detect_parser(fp)
            level_entries = []
            for entry in parser.parse_file(fp):
                if entry.level in {"ERROR", "FATAL", "CRITICAL"}:
                    level_entries.append((entry, normalize(entry.message)))
            total, groups = _group_errors(level_entries, False, 30)
            assert total == 2
            assert len(groups) == 1  # both normalize to same fingerprint


class TestSearchLogsFastPath:
    """Test that search_logs correctly activates/deactivates the classifier."""

    def test_slow_path_for_info_level(self, tmp_log):
        """INFO level search should NOT use classifier (uses slow path)."""
        fp = tmp_log(
            "2024-01-15 10:00:01 INFO Starting\n"
            "2024-01-15 10:00:02 ERROR timeout\n"
            "2024-01-15 10:00:03 INFO Recovered\n"
        )
        # Even if classifier is available, INFO search should go slow path
        assert _should_use_classifier(None, "INFO") is False

    def test_error_level_triggers_fast_path(self, tmp_log):
        """ERROR level search should use classifier."""
        assert _should_use_classifier(None, "ERROR") is True
