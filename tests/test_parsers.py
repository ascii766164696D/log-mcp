from log_mcp.parsing.text_parser import TextLogParser
from log_mcp.parsing.detector import detect_parser


class TestTextLogParser:
    def test_iso_timestamp_with_level(self):
        p = TextLogParser()
        entry = p.parse_line("2024-01-15 10:30:45 ERROR some message", 1)
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.message == "some message"
        assert entry.timestamp is not None
        assert entry.timestamp.year == 2024

    def test_iso_timestamp_with_milliseconds(self):
        p = TextLogParser()
        entry = p.parse_line("2024-01-15 10:30:45.123 INFO starting", 1)
        assert entry is not None
        assert entry.level == "INFO"

    def test_bracketed_timestamp(self):
        p = TextLogParser()
        entry = p.parse_line("[2024-01-15 10:30:45] ERROR something", 1)
        assert entry is not None
        assert entry.level == "ERROR"

    def test_empty_line(self):
        p = TextLogParser()
        assert p.parse_line("", 1) is None
        assert p.parse_line("   ", 1) is None

    def test_locks_to_first_pattern(self):
        p = TextLogParser()
        p.parse_line("2024-01-15 10:30:45 INFO first", 1)
        # Should use the same pattern now
        entry = p.parse_line("2024-01-15 10:30:46 WARN second", 2)
        assert entry is not None
        assert entry.level == "WARN"


class TestDetectParser:
    def test_standard_text_log(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:30:45 INFO Starting\n"
            "2024-01-15 10:30:46 ERROR Failed\n"
        )
        parser = detect_parser(fp)
        assert type(parser).__name__ == "TextLogParser"

    def test_json_log(self, tmp_log):
        fp = tmp_log(
            '{"timestamp": "2024-01-15T10:30:45Z", "level": "INFO", "message": "hi"}\n'
            '{"timestamp": "2024-01-15T10:30:46Z", "level": "ERROR", "message": "fail"}\n'
        )
        parser = detect_parser(fp)
        assert type(parser).__name__ == "JsonLogParser"

    def test_spark_log4j_format(self, tmp_log):
        fp = tmp_log(
            "17/06/08 13:33:49 INFO executor.Backend: Registered\n"
            "17/06/08 13:33:49 INFO spark.SecurityManager: Changing acls\n"
            "17/06/08 13:33:50 WARN something: warning here\n"
        )
        parser = detect_parser(fp)
        assert type(parser).__name__ == "AdaptiveTextParser"
        entries = list(parser.parse_file(fp))
        assert len(entries) == 3
        assert entries[0].level == "INFO"
        assert entries[0].timestamp.year == 2017
        assert entries[2].level == "WARN"

    def test_tab_delimited_ci_log(self, tmp_log):
        fp = tmp_log(
            "test\tUNKNOWN STEP\t2026-02-24T18:21:48.867Z\tStarting runner\n"
            "test\tUNKNOWN STEP\t2026-02-24T18:21:49.000Z\t##[group]Setup\n"
        )
        parser = detect_parser(fp)
        assert type(parser).__name__ == "AdaptiveTextParser"
        entries = list(parser.parse_file(fp))
        assert len(entries) == 2
        assert entries[0].extra.get("prefix") is not None


class TestMultilineEntries:
    def test_continuation_lines_appended(self, tmp_log):
        fp = tmp_log(
            "2024-01-15 10:30:45 ERROR Something failed\n"
            "  at com.example.Foo.bar(Foo.java:42)\n"
            "  at com.example.Main.main(Main.java:10)\n"
            "2024-01-15 10:30:46 INFO Recovered\n"
        )
        parser = detect_parser(fp)
        entries = list(parser.parse_file(fp))
        assert len(entries) == 2
        assert "Foo.bar" in entries[0].message
        assert entries[1].message == "Recovered"
