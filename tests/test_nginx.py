"""Tests for nginx access log and error log parsing."""

from log_mcp.parsing.detector import detect_parser
from log_mcp.tools.errors import register_tools as register_error_tools
from log_mcp.tools.compare import register_tools as register_compare_tools


NGINX_ACCESS_LINES = (
    '172.17.0.1 - - [27/Feb/2026:10:00:00 +0000] "GET /api/health HTTP/1.1" 200 15 "-" "curl/7.88.1"\n'
    '172.17.0.1 - - [27/Feb/2026:10:00:01 +0000] "POST /api/login HTTP/1.1" 401 42 "-" "Mozilla/5.0"\n'
    '10.0.0.5 - admin [27/Feb/2026:10:00:02 +0000] "GET /dashboard HTTP/1.1" 200 1024 "https://example.com" "Mozilla/5.0"\n'
)

NGINX_ERROR_LINES = (
    "2026/02/27 10:00:00 [error] 1234#0: *5 open() \"/var/www/html/missing.html\" failed (2: No such file or directory), client: 172.17.0.1\n"
    "2026/02/27 10:00:01 [error] 1234#0: *6 upstream timed out (110: Connection timed out), client: 10.0.0.5\n"
    "2026/02/27 10:00:02 [warn] 1234#0: conflicting server name \"example.com\", ignored\n"
    "2026/02/27 10:00:03 [info] 1234#0: signal 1 (SIGHUP) received, reconfiguring\n"
)


class TestNginxAccessLog:
    def test_detect_parser(self, tmp_log):
        fp = tmp_log(NGINX_ACCESS_LINES)
        parser = detect_parser(fp)
        assert type(parser).__name__ == "AdaptiveTextParser"

    def test_extract_timestamps(self, tmp_log):
        fp = tmp_log(NGINX_ACCESS_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 3
        assert all(e.timestamp is not None for e in entries)
        assert entries[0].timestamp.year == 2026
        assert entries[0].timestamp.month == 2
        assert entries[0].timestamp.day == 27

    def test_entries_have_messages(self, tmp_log):
        fp = tmp_log(NGINX_ACCESS_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        # Messages should contain the HTTP request details
        assert "GET /api/health" in entries[0].message or "GET /api/health" in entries[0].raw


class TestNginxErrorLog:
    def test_detect_parser(self, tmp_log):
        fp = tmp_log(NGINX_ERROR_LINES)
        parser = detect_parser(fp)
        # Should be detected by adaptive parser (yy/mm/dd format)
        assert type(parser).__name__ in ("AdaptiveTextParser", "TextLogParser")

    def test_extract_timestamps(self, tmp_log):
        fp = tmp_log(NGINX_ERROR_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 4
        assert all(e.timestamp is not None for e in entries)

    def test_extract_levels(self, tmp_log):
        fp = tmp_log(NGINX_ERROR_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert entries[0].level == "ERROR"
        assert entries[1].level == "ERROR"
        assert entries[2].level == "WARN"
        assert entries[3].level == "INFO"


class TestNginxMixedLevels:
    def test_all_nginx_levels(self, tmp_log):
        lines = (
            "2026/02/27 10:00:00 [error] 1#0: something broke\n"
            "2026/02/27 10:00:01 [crit] 1#0: critical issue\n"
            "2026/02/27 10:00:02 [warn] 1#0: watch out\n"
            "2026/02/27 10:00:03 [alert] 1#0: alerting\n"
            "2026/02/27 10:00:04 [notice] 1#0: fyi\n"
            "2026/02/27 10:00:05 [info] 1#0: all good\n"
        )
        fp = tmp_log(lines)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 6
        levels = [e.level for e in entries]
        assert levels[0] == "ERROR"
        assert levels[1] == "CRITICAL"
        assert levels[2] == "WARN"
        assert levels[3] == "FATAL"    # [alert] → FATAL
        assert levels[4] == "NOTICE"   # [notice] stays as NOTICE
        assert levels[5] == "INFO"


class TestNginxAnalyzeErrors:
    def test_finds_errors_in_error_log(self, tmp_log):
        fp = tmp_log(NGINX_ERROR_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        error_entries = [e for e in entries if e.level in ("ERROR", "FATAL", "CRITICAL")]
        assert len(error_entries) == 2  # two [error] lines

    def test_analyze_errors_integration(self, tmp_log):
        """analyze_errors tool should find errors in nginx error logs."""
        from log_mcp.tools.errors import _group_errors
        from log_mcp.normalize import normalize

        fp = tmp_log(NGINX_ERROR_LINES)
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        level_entries = [
            (e, normalize(e.message))
            for e in entries
            if e.level in ("ERROR", "FATAL", "CRITICAL")
        ]
        assert len(level_entries) == 2
        total, groups = _group_errors(level_entries, False, 30)
        assert total == 2
        assert len(groups) >= 1  # at least 1 unique error pattern


class TestNginxCompareLogs:
    def test_compare_two_nginx_files(self, tmp_log):
        """compare_logs should work on two nginx access log files."""
        file1 = tmp_log(
            '172.17.0.1 - - [27/Feb/2026:10:00:00 +0000] "GET /api/health HTTP/1.1" 200 15\n'
            '172.17.0.1 - - [27/Feb/2026:10:00:01 +0000] "GET /api/users HTTP/1.1" 200 100\n',
            name="access1.log",
        )
        file2 = tmp_log(
            '10.0.0.5 - - [27/Feb/2026:10:00:00 +0000] "GET /api/health HTTP/1.1" 200 15\n'
            '10.0.0.5 - - [27/Feb/2026:10:00:01 +0000] "GET /api/admin HTTP/1.1" 403 20\n',
            name="access2.log",
        )
        # Both should parse successfully
        parser1 = detect_parser(file1)
        parser2 = detect_parser(file2)
        entries1 = list(parser1.parse_file(file1))
        entries2 = list(parser2.parse_file(file2))
        assert len(entries1) == 2
        assert len(entries2) == 2

    def test_compare_error_logs(self, tmp_log):
        """compare_logs should work on two nginx error log files."""
        file1 = tmp_log(
            "2026/02/27 10:00:00 [error] 1#0: upstream timed out\n"
            "2026/02/27 10:00:01 [warn] 1#0: conflicting server name\n",
            name="error1.log",
        )
        file2 = tmp_log(
            "2026/02/27 10:00:00 [error] 1#0: upstream timed out\n"
            "2026/02/27 10:00:01 [error] 1#0: open() failed\n",
            name="error2.log",
        )
        parser1 = detect_parser(file1)
        parser2 = detect_parser(file2)
        entries1 = list(parser1.parse_file(file1))
        entries2 = list(parser2.parse_file(file2))
        assert len(entries1) == 2
        assert len(entries2) == 2
        # Both should have levels extracted
        assert entries1[0].level == "ERROR"
        assert entries2[1].level == "ERROR"
