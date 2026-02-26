# log-mcp

MCP server for log file analysis. Gives LLMs the ability to efficiently analyze large log files without loading them into context.

## Tools

| Tool | Description |
|------|-------------|
| `log_overview` | Quick scan: size, line count, time range, level distribution, head/tail samples |
| `search_logs` | Search by regex, log level, and/or time range |
| `get_log_segment` | Extract a segment by line range or time range |
| `analyze_errors` | Deduplicate errors by fingerprint, count frequencies, extract stack traces |
| `log_stats` | Volume histogram, level breakdown, top repeated patterns |
| `compare_logs` | Find patterns unique to each file and frequency outliers across files |

## Key features

- **Auto-detection** of log formats: JSON, standard text (`2024-01-15 10:30:45 ERROR ...`), syslog, Spark/Log4j (`17/06/08 13:33:49 INFO ...`), and tab/pipe-delimited formats (GitHub Actions CI logs)
- **Normalization** collapses variable parts (UUIDs, hex IDs, IPs, numbers) so that messages differing only in IDs or timestamps are grouped as the same pattern
- **Content-based error detection** falls back to regex heuristics (`fatal:`, `Permission denied`, `##[error]`, etc.) when log files lack standard levels
- **Prefix-aware comparison** distinguishes patterns from different job steps in CI logs

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```json
{
  "mcpServers": {
    "log-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/log-mcp", "log-mcp"]
    }
  }
}
```

## Example usage

Analyze errors in a 67MB Spark executor log (705K lines):

```
> analyze_errors("/var/log/spark/container_0002_01_000004.log")

Summary: 34 errors in 5 groups.
Top: 'shuffle.RetryingBlockFetcher: Exception while beginning fetch of <N>
outstanding blocks (after <N> retries) ...' (18x)

--- 18x ---
Fingerprint: shuffle.RetryingBlockFetcher: Exception while beginning fetch of <N> outstanding blocks ...
First: L29764 2017-02-01T15:55:17
Last:  L30677 2017-02-01T15:55:51
Stack trace:
  java.io.IOException: Failed to connect to mesos-slave-13/10.10.34.23:55492
  ...
```

Compare two CI log files:

```
> compare_logs(["run_a.txt", "run_b.txt"])

699 patterns across 2 files (0 shared).
A: 401 unique (top: 'test / test UNKNOWN STEP | ##[endgroup]' 21x)
B: 298 unique (top: 'test UNKNOWN STEP | ##[endgroup]' 21x)
```
