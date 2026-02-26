from log_mcp.normalize import normalize, normalize_extended


class TestNormalize:
    def test_uuid(self):
        assert "<UUID>" in normalize("id=550e8400-e29b-41d4-a716-446655440000")

    def test_bare_hex_40_chars(self):
        """Git SHA (40 hex chars) should be collapsed."""
        sha = "a" * 40
        assert normalize(f"commit {sha}") == "commit <HEX>"

    def test_bare_hex_64_chars(self):
        """Docker container ID (64 hex chars) should be collapsed."""
        docker_id = "abc123def456" * 5 + "abcd"
        assert "<HEX>" in normalize(docker_id)

    def test_bare_hex_12_chars_boundary(self):
        """12-char hex should be collapsed (minimum threshold)."""
        assert normalize("id abc123def456 done") == "id <HEX> done"

    def test_bare_hex_11_chars_not_matched(self):
        """11-char hex should NOT be collapsed."""
        result = normalize("id abc123def45 done")
        assert "<HEX>" not in result

    def test_short_hex_words_not_matched(self):
        """Short English words that are valid hex should not match."""
        result = normalize("the dead beef cafe")
        assert "dead" in result
        assert "beef" in result
        assert "cafe" in result

    def test_0x_prefixed_hex(self):
        assert normalize("addr 0x1a2b3c") == "addr <HEX>"

    def test_numbers(self):
        assert normalize("port 8080 pid 1234") == "port <N> pid <N>"

    def test_id_pattern(self):
        assert normalize("container_abc123") == "<ID>"

    def test_whitespace_collapse(self):
        assert normalize("a  b   c") == "a b c"

    def test_uuid_before_bare_hex(self):
        """UUID should be matched before bare hex to avoid partial matches."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = normalize(f"id={uuid}")
        assert "<UUID>" in result


class TestNormalizeExtended:
    def test_ip_address(self):
        assert "<IP>" in normalize_extended("host 10.0.0.1 port")

    def test_quoted_strings(self):
        result = normalize_extended('key="some value" other')
        assert '"<STR>"' in result

    def test_single_quoted_strings(self):
        result = normalize_extended("key='some value' other")
        assert "'<STR>'" in result

    def test_bare_hex_in_extended(self):
        """Bare hex should also work in extended normalization."""
        sha = "a" * 40
        assert normalize_extended(f"commit {sha}") == "commit <HEX>"
