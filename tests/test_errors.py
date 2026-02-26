from log_mcp.parsing.base import LogEntry
from log_mcp.tools.errors import _group_errors, _CONTENT_ERROR_RE


class TestGroupErrors:
    def _make_entry(self, line: int, msg: str, ts=None):
        return LogEntry(line_number=line, raw=msg, timestamp=ts, message=msg)

    def test_basic_grouping(self):
        entries = [
            (self._make_entry(1, "connection timeout"), "connection timeout"),
            (self._make_entry(5, "connection timeout"), "connection timeout"),
            (self._make_entry(10, "disk full"), "disk full"),
        ]
        total, groups = _group_errors(entries, False, 30)
        assert total == 3
        assert len(groups) == 2
        assert groups["connection timeout"]["count"] == 2
        assert groups["disk full"]["count"] == 1

    def test_first_and_last_seen(self):
        entries = [
            (self._make_entry(1, "err"), "err"),
            (self._make_entry(50, "err"), "err"),
            (self._make_entry(100, "err"), "err"),
        ]
        total, groups = _group_errors(entries, False, 30)
        assert groups["err"]["first_seen"]["line"] == 1
        assert groups["err"]["last_seen"]["line"] == 100

    def test_max_unique_errors(self):
        entries = [
            (self._make_entry(i, f"error_{i}"), f"error_{i}")
            for i in range(10)
        ]
        total, groups = _group_errors(entries, False, 3)
        assert len(groups) == 3
        assert total == 10  # still counts all

    def test_stack_trace_extraction(self):
        raw = "Error happened\n  at Foo.bar(Foo.java:42)\n  at Main.main(Main.java:10)"
        entry = LogEntry(line_number=1, raw=raw, message="Error happened")
        entries = [(entry, "Error happened")]
        _, groups = _group_errors(entries, True, 30)
        assert groups["Error happened"]["stack_trace"] is not None
        assert "Foo.bar" in groups["Error happened"]["stack_trace"]

    def test_no_stack_trace_when_disabled(self):
        raw = "Error happened\n  at Foo.bar(Foo.java:42)"
        entry = LogEntry(line_number=1, raw=raw, message="Error happened")
        entries = [(entry, "Error happened")]
        _, groups = _group_errors(entries, False, 30)
        assert groups["Error happened"]["stack_trace"] is None


class TestContentErrorRegex:
    def test_github_actions_error(self):
        assert _CONTENT_ERROR_RE.search("##[error]Process completed with exit code 1")

    def test_fatal_colon(self):
        assert _CONTENT_ERROR_RE.search("fatal: could not read Username")

    def test_error_colon(self):
        assert _CONTENT_ERROR_RE.search("error: failed to write file")

    def test_permission_denied(self):
        assert _CONTENT_ERROR_RE.search("sudo: unable to execute: Permission denied")

    def test_command_not_found(self):
        assert _CONTENT_ERROR_RE.search("bash: uv: command not found")

    def test_no_such_file(self):
        assert _CONTENT_ERROR_RE.search("No such file or directory")

    def test_exited_with_code(self):
        assert _CONTENT_ERROR_RE.search("Exited with code 1")

    def test_exited_with_code_0_not_matched(self):
        assert not _CONTENT_ERROR_RE.search("Exited with code 0")

    def test_failed(self):
        assert _CONTENT_ERROR_RE.search("Build failed")

    def test_normal_info_not_matched(self):
        assert not _CONTENT_ERROR_RE.search("Starting application server on port 8080")

    def test_could_not(self):
        assert _CONTENT_ERROR_RE.search("Could not resolve host")


class TestAnalyzeErrorsIntegration:
    def test_level_based_detection(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:00:01 INFO Starting\n"
            "2024-01-15 10:00:02 ERROR timeout after 30s\n"
            "2024-01-15 10:00:03 ERROR timeout after 45s\n"
            "2024-01-15 10:00:04 INFO Recovered\n"
        )
        from log_mcp.parsing.detector import detect_parser
        from log_mcp.normalize import normalize

        parser = detect_parser(fp)
        level_entries = []
        for entry in parser.parse_file(fp):
            if entry.level in {"ERROR", "FATAL", "CRITICAL"}:
                level_entries.append((entry, normalize(entry.message)))

        total, groups = _group_errors(level_entries, False, 30)
        assert total == 2
        assert len(groups) == 1  # both timeouts normalize to same fingerprint

    def test_heuristic_fallback(self, tmp_log):
        """When no standard levels exist, content heuristics should find errors."""
        fp = tmp_log(
            "deploy\t2026-02-24T10:00:00Z\tStarting deploy\n"
            "deploy\t2026-02-24T10:00:01Z\tfatal: could not read Username\n"
            "deploy\t2026-02-24T10:00:02Z\tPermission denied\n"
            "deploy\t2026-02-24T10:00:03Z\tDeploy complete\n"
        )
        from log_mcp.parsing.detector import detect_parser
        from log_mcp.normalize import normalize

        parser = detect_parser(fp)
        all_entries = list(parser.parse_file(fp))

        # No level-based errors
        level_entries = [
            (e, normalize(e.message))
            for e in all_entries
            if e.level in {"ERROR", "FATAL", "CRITICAL"}
        ]
        assert len(level_entries) == 0

        # Heuristic fallback should find 2
        content_entries = [
            (e, normalize(e.message))
            for e in all_entries
            if _CONTENT_ERROR_RE.search(e.message)
        ]
        assert len(content_entries) == 2
