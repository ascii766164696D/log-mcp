from __future__ import annotations

import string

from mcp.server.fastmcp import FastMCP

from ..normalize import normalize
from ..parsing.detector import detect_parser
from ..util import validate_file

_LABELS = list(string.ascii_uppercase)  # A, B, C, ...


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def compare_logs(
        file_paths: list[str],
        max_unique_per_file: int = 50,
        max_shared_patterns: int = 20,
        frequency_ratio_threshold: float = 2.0,
    ) -> str:
        """Compare multiple log files and find entries unique to each file.

        Normalises variable parts (numbers, UUIDs, hex) so that messages differing
        only in IDs or timestamps are treated as the same pattern. Returns patterns
        that appear in some files but not in others, helping you focus on what is
        different rather than what is common. Also shows shared patterns and
        frequency outliers where the same pattern appears with very different
        counts across files.
        """
        if len(file_paths) < 2:
            return "Error: Provide at least 2 file paths to compare."

        # Validate all files first
        for fp in file_paths:
            err = validate_file(fp)
            if err:
                return f"Error: {err}"

        # Short labels for files
        labels = _LABELS[: len(file_paths)]
        files_legend = {label: fp for label, fp in zip(labels, file_paths)}

        # Build per-file sets of normalised message patterns
        file_patterns: list[dict[str, dict]] = []  # pattern -> {level, count}
        for fp in file_paths:
            parser = detect_parser(fp)
            patterns: dict[str, dict] = {}
            for entry in parser.parse_file(fp):
                prefix = entry.extra.get("prefix")
                norm = normalize(entry.message)
                if prefix:
                    norm = normalize(prefix) + " | " + norm
                if norm not in patterns:
                    patterns[norm] = {
                        "level": entry.level,
                        "count": 0,
                    }
                patterns[norm]["count"] += 1
            file_patterns.append(patterns)

        all_pattern_keys = set()
        for patterns in file_patterns:
            all_pattern_keys.update(patterns.keys())

        # Shared patterns: present in every file
        shared = all_pattern_keys.copy()
        for patterns in file_patterns:
            shared &= set(patterns.keys())

        # Per-file unique patterns: present in this file but no other
        per_file_unique: list[tuple[str, int, list[tuple[int, str, str]]]] = []
        for idx, (label, patterns) in enumerate(zip(labels, file_patterns)):
            other_keys: set[str] = set()
            for j, p in enumerate(file_patterns):
                if j != idx:
                    other_keys.update(p.keys())

            unique_keys = set(patterns.keys()) - other_keys
            unique_entries = sorted(
                [
                    (patterns[k]["count"], patterns[k]["level"] or "", k)
                    for k in unique_keys
                ],
                key=lambda e: e[0],
                reverse=True,
            )[:max_unique_per_file]

            per_file_unique.append((label, len(unique_keys), unique_entries))

        # Build shared_patterns detail list
        shared_detail: list[tuple[str, str, dict[str, int]]] = []
        for key in shared:
            counts = {
                label: file_patterns[idx][key]["count"]
                for idx, label in enumerate(labels)
            }
            level = file_patterns[0][key]["level"] or ""
            total_count = sum(counts.values())
            shared_detail.append((key, level, counts))
        shared_detail.sort(key=lambda e: sum(e[2].values()), reverse=True)
        shared_detail = shared_detail[:max_shared_patterns]

        # Build frequency_outliers: shared patterns with skewed counts
        outliers: list[tuple[float, str, str, dict[str, int]]] = []
        for key in shared:
            counts_list = [file_patterns[idx][key]["count"] for idx in range(len(file_paths))]
            min_count = min(counts_list)
            max_count = max(counts_list)
            if min_count > 0:
                ratio = max_count / min_count
            else:
                ratio = float(max_count) if max_count > 0 else 1.0
            if ratio >= frequency_ratio_threshold:
                counts = {
                    label: file_patterns[idx][key]["count"]
                    for idx, label in enumerate(labels)
                }
                level = file_patterns[0][key]["level"] or ""
                outliers.append((round(ratio, 1), key, level, counts))
        outliers.sort(key=lambda e: e[0], reverse=True)

        # Build summary
        summary_parts = [
            f"{len(all_pattern_keys)} patterns across {len(file_paths)} files"
            f" ({len(shared)} shared)."
        ]
        for label, unique_count, unique_entries in per_file_unique:
            if unique_entries:
                top_count, top_level, top_pattern = unique_entries[0]
                error_tag = ""
                if top_level in {"ERROR", "FATAL", "CRITICAL"}:
                    error_tag = " [error]"
                summary_parts.append(
                    f"{label}: {unique_count} unique"
                    f" (top: '{top_pattern}' {top_count}x{error_tag})"
                )
        if outliers:
            o = outliers[0]
            summary_parts.append(
                f"Biggest outlier: '{o[1]}' ({o[0]}x ratio)"
            )

        # Build plain text output
        parts: list[str] = []
        files_str = " ".join(f"{l}={fp}" for l, fp in files_legend.items())
        parts.append(f"Files: {files_str}")
        parts.append(f"Summary: {' '.join(summary_parts)}")

        for label, unique_count, unique_entries in per_file_unique:
            if unique_entries:
                parts.append("")
                parts.append(f"=== Unique to {label} ({unique_count} total) ===")
                for count, level, pattern in unique_entries:
                    level_tag = f"[{level}] " if level else ""
                    parts.append(f"  {count:>4}x {level_tag}{pattern}")

        if shared_detail:
            total_shared = len(shared)
            parts.append("")
            if len(shared_detail) < total_shared:
                parts.append(f"=== Shared (top {len(shared_detail)} of {total_shared}) ===")
            else:
                parts.append(f"=== Shared ({total_shared}) ===")
            for pattern, level, counts in shared_detail:
                counts_str = " ".join(f"{l}={c}" for l, c in counts.items())
                level_tag = f"[{level}] " if level else ""
                parts.append(f"  {counts_str} {level_tag}{pattern}")

        if outliers:
            parts.append("")
            parts.append(f"=== Frequency outliers (ratio >= {frequency_ratio_threshold}) ===")
            for ratio, pattern, level, counts in outliers:
                counts_str = " ".join(f"{l}={c}" for l, c in counts.items())
                level_tag = f"[{level}] " if level else ""
                parts.append(f"  {ratio}x {counts_str} {level_tag}{pattern}")

        return "\n".join(parts)
