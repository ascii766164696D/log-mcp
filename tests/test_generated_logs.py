"""Integration tests using scripts/generate_test_logs.py to produce realistic log data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_test_logs import (
    generate_log,
    redis_cache_failure,
    null_pointer_error,
    validation_error,
    payment_timeout,
    webhook_signature_error,
    rate_limit_exceeded,
    upstream_error,
)

from log_mcp.parsing.detector import detect_parser
from log_mcp.parsing.base import count_lines
from log_mcp.normalize import normalize
from log_mcp.tools.errors import _group_errors, _ERROR_LEVELS, _CONTENT_ERROR_RE


# ---- Fixtures ------------------------------------------------------------ #

SERVER1_SHARED = {
    "health_check": 8,
    "get_orders": 10,
    "post_orders": 12,
    "get_users": 5,
    "user_login": 4,
    "user_logout": 3,
    "get_inventory": 4,
    "delete_order": 2,
    "health_check_ok": 8,
    "deprecated_api": 3,
    "cache_hit_ratio": 2,
    "scheduled_job": 2,
    "rate_limit_warn": 2,
    "inventory_low": 2,
    "conn_pool_exhausted": 2,
}

SERVER2_SHARED = {
    "health_check": 18,
    "get_orders": 20,
    "post_orders": 5,
    "get_users": 5,
    "user_login": 4,
    "user_logout": 3,
    "get_inventory": 4,
    "delete_order": 2,
    "health_check_ok": 18,
    "deprecated_api": 3,
    "cache_hit_ratio": 2,
    "scheduled_job": 2,
    "rate_limit_warn": 3,
    "inventory_low": 2,
    "conn_pool_exhausted": 1,
}

SERVER1_UNIQUE = [
    (4, redis_cache_failure),
    (3, None),   # gc_pause – we import lazily below
    (3, None),   # slow_query_standalone
    (3, None),   # memory_warning
    (1.5, null_pointer_error),
    (1.5, validation_error),
    (1, None),   # redis_reconnected
]

SERVER2_UNIQUE = [
    (3, payment_timeout),
    (4, webhook_signature_error),
    (2, rate_limit_exceeded),
    (2, None),   # cert_warning
    (2, upstream_error),
]


@pytest.fixture(scope="module")
def generated_logs(tmp_path_factory):
    """Generate both log files once for the whole module."""
    import random
    import importlib
    import generate_test_logs as gen

    # Reset seed for deterministic output
    random.seed(42)

    log1 = gen.generate_log(
        target_lines=1500,
        port=8080,
        db_host="db-primary.internal",
        cache_hosts=["redis-01.internal", "redis-02.internal"],
        ip_prefix="192.168",
        shared_weights=SERVER1_SHARED,
        unique_generators=[
            (4, gen.redis_cache_failure),
            (3, gen.gc_pause),
            (3, gen.slow_query_standalone),
            (3, gen.memory_warning),
            (1.5, gen.null_pointer_error),
            (1.5, gen.validation_error),
            (1, gen.redis_reconnected),
        ],
    )

    log2 = gen.generate_log(
        target_lines=1500,
        port=9090,
        db_host="db-replica.internal",
        cache_hosts=["redis-03.internal", "redis-04.internal"],
        ip_prefix="10.0",
        shared_weights=SERVER2_SHARED,
        unique_generators=[
            (3, gen.payment_timeout),
            (4, gen.webhook_signature_error),
            (2, gen.rate_limit_exceeded),
            (2, gen.cert_warning),
            (2, gen.upstream_error),
        ],
    )

    d = tmp_path_factory.mktemp("generated")
    fp1 = str(d / "server1.log")
    fp2 = str(d / "server2.log")
    Path(fp1).write_text("\n".join(log1) + "\n")
    Path(fp2).write_text("\n".join(log2) + "\n")

    return fp1, fp2


# ---- Parsing tests ------------------------------------------------------- #

class TestGeneratedParsing:
    def test_parser_detects_text_format(self, generated_logs):
        fp1, fp2 = generated_logs
        p1 = detect_parser(fp1)
        p2 = detect_parser(fp2)
        assert type(p1).__name__ == "TextLogParser"
        assert type(p2).__name__ == "TextLogParser"

    def test_line_count_matches(self, generated_logs):
        fp1, fp2 = generated_logs
        assert count_lines(fp1) >= 1500
        assert count_lines(fp2) >= 1500

    def test_all_entries_have_timestamps(self, generated_logs):
        """Every generated entry starts with a timestamp, so the parser should find one."""
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        entries = list(parser.parse_file(fp1))
        # Multiline continuation lines get merged, so all entries should have timestamps
        for entry in entries:
            assert entry.timestamp is not None, f"Missing timestamp: {entry.raw[:80]}"

    def test_level_distribution(self, generated_logs):
        """Generated logs should contain DEBUG, INFO, WARN, and ERROR levels."""
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        levels = set()
        for entry in parser.parse_file(fp1):
            if entry.level:
                levels.add(entry.level)
        assert {"DEBUG", "INFO", "WARN", "ERROR"}.issubset(levels)


# ---- Error analysis tests ------------------------------------------------ #

class TestGeneratedErrors:
    def test_server1_has_errors(self, generated_logs):
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        error_entries = []
        for entry in parser.parse_file(fp1):
            if entry.level in _ERROR_LEVELS:
                error_entries.append((entry, normalize(entry.message)))

        total, groups = _group_errors(error_entries, False, 30)
        assert total > 0, "Server1 should have errors"
        assert len(groups) >= 3, "Server1 should have at least 3 error types (redis, null pointer, validation)"

    def test_server2_has_errors(self, generated_logs):
        _, fp2 = generated_logs
        parser = detect_parser(fp2)
        error_entries = []
        for entry in parser.parse_file(fp2):
            if entry.level in _ERROR_LEVELS:
                error_entries.append((entry, normalize(entry.message)))

        total, groups = _group_errors(error_entries, False, 30)
        assert total > 0, "Server2 should have errors"
        assert len(groups) >= 3, "Server2 should have at least 3 error types (payment, webhook, rate limit)"

    def test_server1_error_fingerprints(self, generated_logs):
        """Server1 errors should include redis cache and null pointer patterns."""
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        error_entries = []
        for entry in parser.parse_file(fp1):
            if entry.level in _ERROR_LEVELS:
                error_entries.append((entry, normalize(entry.message)))

        _, groups = _group_errors(error_entries, False, 30)
        fingerprints = set(groups.keys())
        assert any("cache" in fp.lower() for fp in fingerprints), \
            f"Expected redis cache errors, got: {fingerprints}"
        assert any("null" in fp.lower() or "nonetype" in fp.lower() for fp in fingerprints), \
            f"Expected null pointer errors, got: {fingerprints}"

    def test_server2_error_fingerprints(self, generated_logs):
        """Server2 errors should include payment timeout and webhook patterns."""
        _, fp2 = generated_logs
        parser = detect_parser(fp2)
        error_entries = []
        for entry in parser.parse_file(fp2):
            if entry.level in _ERROR_LEVELS:
                error_entries.append((entry, normalize(entry.message)))

        _, groups = _group_errors(error_entries, False, 30)
        fingerprints = set(groups.keys())
        assert any("timeout" in fp.lower() or "payment" in fp.lower() for fp in fingerprints), \
            f"Expected payment timeout errors, got: {fingerprints}"
        assert any("webhook" in fp.lower() or "signature" in fp.lower() for fp in fingerprints), \
            f"Expected webhook signature errors, got: {fingerprints}"

    def test_stack_trace_extraction(self, generated_logs):
        """Server1 redis/null errors have multiline stack traces; they should be captured."""
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        error_entries = []
        for entry in parser.parse_file(fp1):
            if entry.level in _ERROR_LEVELS:
                error_entries.append((entry, normalize(entry.message)))

        _, groups = _group_errors(error_entries, True, 30)
        has_trace = any(g["stack_trace"] is not None for g in groups.values())
        assert has_trace, "At least one error group should have a stack trace"


# ---- Compare tests ------------------------------------------------------- #

class TestGeneratedCompare:
    def _build_patterns(self, fp: str) -> dict[str, int]:
        parser = detect_parser(fp)
        patterns: dict[str, int] = {}
        for entry in parser.parse_file(fp):
            norm = normalize(entry.message)
            patterns[norm] = patterns.get(norm, 0) + 1
        return patterns

    def test_shared_patterns_exist(self, generated_logs):
        fp1, fp2 = generated_logs
        p1 = self._build_patterns(fp1)
        p2 = self._build_patterns(fp2)
        shared = set(p1.keys()) & set(p2.keys())
        # Health check, get orders, etc. should appear in both
        assert len(shared) >= 5, f"Expected many shared patterns, got {len(shared)}"

    def test_unique_patterns_per_server(self, generated_logs):
        fp1, fp2 = generated_logs
        p1 = self._build_patterns(fp1)
        p2 = self._build_patterns(fp2)
        only_s1 = set(p1.keys()) - set(p2.keys())
        only_s2 = set(p2.keys()) - set(p1.keys())
        assert len(only_s1) > 0, "Server1 should have unique patterns"
        assert len(only_s2) > 0, "Server2 should have unique patterns"

    def test_redis_patterns_unique_to_server1(self, generated_logs):
        fp1, fp2 = generated_logs
        p1 = self._build_patterns(fp1)
        p2 = self._build_patterns(fp2)
        only_s1 = set(p1.keys()) - set(p2.keys())
        assert any("cache" in pat.lower() for pat in only_s1), \
            f"Redis cache patterns should be unique to server1, unique: {[p for p in only_s1 if 'cache' in p.lower() or 'redis' in p.lower()]}"

    def test_payment_patterns_unique_to_server2(self, generated_logs):
        fp1, fp2 = generated_logs
        p1 = self._build_patterns(fp1)
        p2 = self._build_patterns(fp2)
        only_s2 = set(p2.keys()) - set(p1.keys())
        assert any("payment" in pat.lower() or "webhook" in pat.lower() for pat in only_s2), \
            f"Payment/webhook patterns should be unique to server2, unique: {[p for p in only_s2 if 'payment' in p.lower() or 'webhook' in p.lower()]}"

    def test_frequency_outliers(self, generated_logs):
        """Health checks and GET orders have 2x weight on server2 — should produce outliers."""
        fp1, fp2 = generated_logs
        p1 = self._build_patterns(fp1)
        p2 = self._build_patterns(fp2)
        shared = set(p1.keys()) & set(p2.keys())

        outliers = []
        for key in shared:
            c1, c2 = p1[key], p2[key]
            min_c = min(c1, c2)
            if min_c > 0:
                ratio = max(c1, c2) / min_c
                if ratio >= 2.0:
                    outliers.append((ratio, key, c1, c2))

        assert len(outliers) > 0, "Should have frequency outliers for health_check/get_orders patterns"

    def test_normalization_collapses_ids(self, generated_logs):
        """Order IDs, user IDs, IP addresses should be normalized so patterns collapse."""
        fp1, _ = generated_logs
        parser = detect_parser(fp1)
        raw_messages = set()
        norm_messages = set()
        for entry in parser.parse_file(fp1):
            raw_messages.add(entry.message)
            norm_messages.add(normalize(entry.message))
        # Normalization should significantly reduce cardinality
        assert len(norm_messages) < len(raw_messages), \
            f"Normalization should collapse patterns: {len(raw_messages)} raw vs {len(norm_messages)} normalized"
