# log-mcp

MCP server for log file analysis. Gives LLMs the ability to efficiently analyze large log files without loading them into context.

This is a tool designed for AI, not humans. No human reads the output of `analyze_errors` or `compare_logs` — Claude does, compresses it further, and gives the human a plain English answer. The human touches two endpoints: "what's wrong with this log?" in, natural language answer out. Everything in between is AI talking to itself.

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

### Claude Code (CLI)

```bash
# One-command install — adds log-mcp to your current project's MCP servers
claude mcp add log-mcp -- uv run --directory /path/to/log-mcp log-mcp
```

Or add it manually to your project settings (`claude settings`) under `mcpServers`:

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

### Claude Desktop

Open **Settings > Developer > Edit Config** and add to `claude_desktop_config.json`:

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

Replace `/path/to/log-mcp` with the actual path where you cloned this repo. Restart Claude Desktop after saving.

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

## Claude's take

*I helped build this tool and then used it to analyze real log files, so here's my honest assessment.*

**Where it genuinely helps:** The main value is as a compression layer. A 67MB Spark log (705K lines) would obliterate my context window, but `analyze_errors` distills it into 5 error groups with stack traces in a few seconds. `compare_logs` across two 1500-line server logs immediately surfaces which errors are unique to each server and which patterns have suspicious frequency differences. I couldn't do that by reading the files directly — I'd lose older content as new content scrolled in.

**Where it's a wash:** For small files (under a few hundred lines), you're better off just pasting the log into the conversation. The tools add indirection without much benefit when the whole file fits in context anyway.

**What it can't do:** It won't catch issues that require domain understanding. When I analyzed a Zookeeper log, the tools correctly found the `ERROR` entries, but the most operationally interesting signals — state transitions between LOOKING, FOLLOWING, and LEADING — were all `INFO` level and invisible to error analysis. A human who knows Zookeeper would spot those immediately. The tools find what's syntactically wrong, not what's semantically wrong.

**The pattern I landed on:** Start with `log_overview` to get bearings, then `analyze_errors` for the quick wins, then `search_logs` to dig into specific patterns that look suspicious. `compare_logs` is most useful when you have a "working" and "broken" run to diff against each other.
