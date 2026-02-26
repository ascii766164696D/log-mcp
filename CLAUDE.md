# CLAUDE.md

## Project overview

log-mcp is an MCP server that gives LLMs tools for analyzing log files without loading them into context. It auto-detects log formats, normalizes variable parts for pattern matching, and provides overview/search/compare/error-analysis tools.

## Build & test

```bash
uv run pytest           # run all tests
uv run pytest -v        # verbose
uv run log-mcp          # start the MCP server (stdio transport)
```

No build step — `uv run` handles everything via `pyproject.toml`.

## Architecture

```
src/log_mcp/
  parsing/          # Format detection + parsers (JSON, text, adaptive)
  tools/            # One module per MCP tool (overview, search, segment, stats, errors, compare)
  normalize.py      # Pattern normalization (UUIDs, hex, IPs, numbers → placeholders)
  server.py         # FastMCP server setup, registers all tools
  util.py           # Shared helpers (file validation, time parsing)
```

Detection order: JSON → TextLogParser (hardcoded patterns) → AdaptiveTextParser (timestamp anchor discovery) → fallback.

## What could be improved

These are genuine gaps I noticed while building and testing against real log files — not speculative wishlist items.

### Parser gaps

- **Zookeeper/Log4j embedded levels.** Lines like `2017-... - INFO [main:ZooKeeper@...] - Session established` have the level buried inside the line after a ` - ` separator. The text parser doesn't recognize this; all entries get `level=None`. The adaptive parser could learn this pattern but currently doesn't look for ` - LEVEL [` anchors.

- **TextLogParser pattern locking is fragile.** It locks to the first regex that matches and uses it for the entire file. If a file has a preamble (config dump, banner) before the actual log lines start, it may lock to the wrong pattern and miss everything else.

- **No logfmt support.** `key=value key2=value2` structured logs (common in Go services) aren't recognized. Would need a dedicated parser.

- **No XML/HTML log support.** Log4j XML appender output, Windows Event Log exports — these exist in the wild but aren't handled.

### Normalization gaps

- **Bare hex minimum threshold (12 chars) is a heuristic.** It avoids false positives on English words like "dead" and "cafe", but it means short hex tokens (8-11 chars) in things like short container IDs survive normalization. Lowering the threshold would create false positives; raising it would miss more IDs.

- **Paths and URLs aren't normalized.** `/var/log/app-server-03/output.log` and `/var/log/app-server-04/output.log` produce different patterns. Could add a path normalizer, but it's tricky to avoid over-collapsing.

- **normalize vs normalize_extended inconsistency.** `compare_logs` uses base `normalize()` while `log_stats` uses `normalize_extended()`. This means the same message can produce different fingerprints depending on which tool you use. Should probably standardize, but `normalize_extended` is more aggressive (collapses quoted strings) which may over-collapse for comparison purposes.

### Tool gaps

- **`analyze_errors` two-pass is all-or-nothing.** If the file has _any_ entries with `ERROR`/`FATAL`/`CRITICAL` level, the content heuristic pass is skipped entirely. A file could have 2 level-based errors and 50 `fatal:` messages in level-less entries, and the 50 would be invisible.

- **`compare_logs` prefix handling can fragment patterns.** When comparing CI logs with different job step names (e.g., `test / test` vs `test`), the prefix gets prepended to each pattern key. If the same log message appears under different prefixes, it shows as "unique to each file" instead of "shared." This made the GH Actions comparison show 0 shared patterns.

- **`log_overview` reads the file twice.** Once for raw head/tail lines, once for parsed entries. For very large files over a network mount this is wasteful. Could collect both in a single pass.

- **No streaming/pagination for large results.** `search_logs` has `max_results` but other tools (`log_stats`, `compare_logs`) scan the entire file and build everything in memory. For multi-GB files with millions of entries, this could be slow or OOM.

- **Time range filtering in `search_logs` doesn't short-circuit.** If the file is sorted by time and the query has an `end_time`, the search continues scanning the entire file even after passing the end boundary.

### Testing gaps

- **No test for JSON log parsing end-to-end.** `test_parsers.py` checks detection but doesn't verify that JsonLogParser correctly extracts timestamps, levels, and messages from realistic JSON log entries.

- **No test for malformed/mixed-format files.** Real log files often have corrupted lines, encoding issues, or format changes mid-file. No tests verify graceful degradation.

- **No benchmarks.** No tests that verify performance on large files. The tools are designed for large files but we only test with small synthetic data.

### Infrastructure

- **No CI pipeline.** Tests exist but aren't run automatically on push.

- **No type checking.** The code uses type hints but `mypy`/`pyright` isn't configured or run.

- **Server restart required after code changes.** The MCP server runs as a long-lived process. Code changes require killing and restarting — there's no hot-reload, which makes development iteration slow.
