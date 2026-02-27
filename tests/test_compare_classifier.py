"""Tests for LOOK/SKIP scoring integration in compare_logs."""

from __future__ import annotations

from unittest.mock import patch

from log_mcp.tools.compare import register_tools


def _make_tool(mcp_mock):
    """Register tools and return the compare_logs function."""
    captured = {}

    class FakeMCP:
        def tool(self):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn
            return decorator

    fake = FakeMCP()
    register_tools(fake)
    return captured["compare_logs"]


class _FakeClassifier:
    """Mock classifier that assigns high probability to ERROR/FATAL patterns."""

    def classify_batch(self, lines: list[str], threshold: float):
        results = []
        for line in lines:
            upper = line.upper()
            if any(kw in upper for kw in ("ERROR", "FATAL", "CRITICAL", "EXCEPTION")):
                results.append(("LOOK", 0.95))
            elif any(kw in upper for kw in ("WARN",)):
                results.append(("LOOK", 0.60))
            else:
                results.append(("SKIP", 0.10))
        return results


class TestClassifierSorting:
    def test_unique_patterns_sorted_by_probability(self, tmp_log):
        """Error patterns should sort above info patterns even with lower count."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 100
            + "2024-01-15 10:30:46 ERROR connection refused\n",
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO startup complete\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=_FakeClassifier()):
            result = compare_logs(file_paths=[file_a, file_b])

        lines = result.split("\n")
        unique_a_section = []
        in_section = False
        for line in lines:
            if "Unique to A" in line:
                in_section = True
                continue
            elif line.startswith("===") and in_section:
                break
            elif in_section and line.strip():
                unique_a_section.append(line)

        # ERROR pattern should be first despite having count=1 vs heartbeat count=100
        assert len(unique_a_section) >= 2
        assert "connection refused" in unique_a_section[0]
        assert "heartbeat" in unique_a_section[1]

    def test_probability_tags_in_output(self, tmp_log):
        """Output should contain [prob] tags when classifier is available."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 ERROR something broke\n",
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO all good\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=_FakeClassifier()):
            result = compare_logs(file_paths=[file_a, file_b])

        # Should have [0.95] tag for the error pattern
        assert "[0.95]" in result
        # Should have [0.10] tag for the info pattern
        assert "[0.10]" in result

    def test_no_probability_tags_without_classifier(self, tmp_log):
        """Output should NOT contain [prob] tags when classifier is unavailable."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 ERROR something broke\n",
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO all good\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=None):
            result = compare_logs(file_paths=[file_a, file_b])

        # No probability tags should appear
        import re
        assert not re.search(r"\[\d+\.\d+\]", result)

    def test_fallback_sorts_by_count(self, tmp_log):
        """Without classifier, patterns should sort by count (original behavior)."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 50
            + "2024-01-15 10:30:46 ERROR connection refused\n",
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO startup complete\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=None):
            result = compare_logs(file_paths=[file_a, file_b])

        lines = result.split("\n")
        unique_a_section = []
        in_section = False
        for line in lines:
            if "Unique to A" in line:
                in_section = True
                continue
            elif line.startswith("===") and in_section:
                break
            elif in_section and line.strip():
                unique_a_section.append(line)

        # Without classifier, heartbeat (50x) should sort before error (1x)
        assert len(unique_a_section) >= 2
        assert "heartbeat" in unique_a_section[0]
        assert "connection refused" in unique_a_section[1]

    def test_shared_patterns_sorted_by_probability(self, tmp_log):
        """Shared patterns should sort by LOOK probability."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 100
            + "2024-01-15 10:30:46 ERROR timeout reached\n",
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 100
            + "2024-01-15 10:30:46 ERROR timeout reached\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=_FakeClassifier()):
            result = compare_logs(file_paths=[file_a, file_b])

        lines = result.split("\n")
        shared_section = []
        in_section = False
        for line in lines:
            if "=== Shared" in line:
                in_section = True
                continue
            elif line.startswith("===") and in_section:
                break
            elif in_section and line.strip():
                shared_section.append(line)

        # ERROR pattern should be first despite lower count
        assert len(shared_section) >= 2
        assert "timeout" in shared_section[0]
        assert "heartbeat" in shared_section[1]

    def test_outliers_sorted_by_probability(self, tmp_log):
        """Outlier patterns should sort by LOOK probability."""
        file_a = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 100
            + "2024-01-15 10:30:46 ERROR timeout reached\n" * 5,
            name="a.log",
        )
        file_b = tmp_log(
            "2024-01-15 10:30:45 INFO heartbeat ok\n" * 10
            + "2024-01-15 10:30:46 ERROR timeout reached\n",
            name="b.log",
        )

        compare_logs = _make_tool(None)

        with patch("log_mcp.tools.compare.get_classifier", return_value=_FakeClassifier()):
            result = compare_logs(
                file_paths=[file_a, file_b],
                frequency_ratio_threshold=2.0,
            )

        lines = result.split("\n")
        outlier_section = []
        in_section = False
        for line in lines:
            if "Frequency outliers" in line:
                in_section = True
                continue
            elif line.startswith("===") and in_section:
                break
            elif in_section and line.strip():
                outlier_section.append(line)

        # Both are outliers (ratio >= 2); error should be first
        if len(outlier_section) >= 2:
            assert "timeout" in outlier_section[0]
            assert "heartbeat" in outlier_section[1]
