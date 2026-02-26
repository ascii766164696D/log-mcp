from log_mcp.normalize import normalize
from log_mcp.parsing.detector import detect_parser


class TestComparePrefix:
    def test_prefix_included_in_pattern_key(self, tmp_log):
        """When entries have a prefix, it should be prepended to the normalized pattern."""
        fp = tmp_log(
            "deploy\t2026-02-24T10:00:00Z\tStarting service\n"
            "test\t2026-02-24T10:00:01Z\tStarting service\n"
        )
        parser = detect_parser(fp)
        patterns = {}
        for entry in parser.parse_file(fp):
            prefix = entry.extra.get("prefix")
            norm = normalize(entry.message)
            if prefix:
                norm = normalize(prefix) + " | " + norm
            patterns[norm] = patterns.get(norm, 0) + 1

        # Same message but different prefix → different keys
        assert len(patterns) == 2
        assert any("deploy" in k for k in patterns)
        assert any("test" in k for k in patterns)

    def test_no_prefix_for_standard_logs(self, tmp_log):
        """Standard text logs should not have a prefix."""
        fp = tmp_log(
            "2024-01-15 10:30:45 INFO Starting\n"
            "2024-01-15 10:30:46 INFO Starting\n"
        )
        parser = detect_parser(fp)
        for entry in parser.parse_file(fp):
            assert "prefix" not in entry.extra


class TestCompareHexNormalization:
    def test_docker_ids_collapsed(self, tmp_log):
        """Docker layer IDs should collapse to the same pattern."""
        sha1 = "a" * 64
        sha2 = "b" * 64
        fp = tmp_log(
            f"2024-01-15 10:30:45 INFO {sha1}: Pulling\n"
            f"2024-01-15 10:30:46 INFO {sha2}: Pulling\n"
        )
        parser = detect_parser(fp)
        patterns = set()
        for entry in parser.parse_file(fp):
            patterns.add(normalize(entry.message))

        assert len(patterns) == 1
        assert "<HEX>: Pulling" in patterns.pop()

    def test_git_shas_collapsed(self, tmp_log):
        sha1 = "abc123" * 6 + "abcd"  # 40 chars
        sha2 = "def456" * 6 + "def4"
        fp = tmp_log(
            f"2024-01-15 10:30:45 INFO Commit {sha1}\n"
            f"2024-01-15 10:30:46 INFO Commit {sha2}\n"
        )
        parser = detect_parser(fp)
        patterns = set()
        for entry in parser.parse_file(fp):
            patterns.add(normalize(entry.message))

        assert len(patterns) == 1
