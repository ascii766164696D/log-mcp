"""Integration tests that exercise the full tool code paths via direct function calls."""

from log_mcp.parsing.detector import detect_parser
from log_mcp.parsing.base import count_lines


class TestLogOverviewIntegration:
    def test_overview_standard_log(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:00:00 INFO Starting\n"
            "2024-01-15 10:00:01 WARN Low memory\n"
            "2024-01-15 10:00:02 ERROR Crash\n"
            "2024-01-15 10:00:03 INFO Recovered\n"
        )
        assert count_lines(fp) == 4
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 4
        levels = {e.level for e in entries}
        assert levels == {"INFO", "WARN", "ERROR"}

    def test_overview_spark_log(self, tmp_log):
        fp = tmp_log(
            "17/06/08 13:33:49 INFO executor.Backend: Registered\n"
            "17/06/08 13:33:50 WARN spark.Security: Something\n"
        )
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 2
        assert entries[0].timestamp.year == 2017
        assert entries[0].level == "INFO"
        assert entries[1].level == "WARN"


class TestSearchIntegration:
    def test_search_by_level(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:00:00 INFO Starting\n"
            "2024-01-15 10:00:01 ERROR Failed\n"
            "2024-01-15 10:00:02 ERROR Timeout\n"
            "2024-01-15 10:00:03 INFO Done\n"
        )
        parser = detect_parser(fp)
        errors = [e for e in parser.parse_file(fp) if e.level == "ERROR"]
        assert len(errors) == 2

    def test_search_by_pattern(self, tmp_log):
        import re
        fp = tmp_log(
            "2024-01-15 10:00:00 INFO Starting app v1.2.3\n"
            "2024-01-15 10:00:01 INFO Connected to db\n"
            "2024-01-15 10:00:02 ERROR Connection timeout to db\n"
        )
        parser = detect_parser(fp)
        regex = re.compile("db", re.IGNORECASE)
        matches = [e for e in parser.parse_file(fp) if regex.search(e.raw)]
        assert len(matches) == 2


class TestStatsIntegration:
    def test_level_counts(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:00:00 INFO a\n"
            "2024-01-15 10:00:01 INFO b\n"
            "2024-01-15 10:00:02 WARN c\n"
            "2024-01-15 10:00:03 ERROR d\n"
        )
        parser = detect_parser(fp)
        levels = {}
        for entry in parser.parse_file(fp):
            if entry.level:
                levels[entry.level] = levels.get(entry.level, 0) + 1
        assert levels == {"INFO": 2, "WARN": 1, "ERROR": 1}


class TestGzipSupport:
    def test_gzip_file(self, tmp_path):
        import gzip
        fp = str(tmp_path / "test.log.gz")
        with gzip.open(fp, "wt") as f:
            f.write("2024-01-15 10:00:00 INFO Starting\n")
            f.write("2024-01-15 10:00:01 ERROR Failed\n")

        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 2
        assert count_lines(fp) == 2
