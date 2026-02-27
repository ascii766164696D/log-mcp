"""Sample LOOK lines that don't contain ERROR or WARN keywords.

Reads a small sample from each dataset, runs the model, and prints
LOOK-classified lines that have no ERROR/WARN keywords — to understand
what else the model considers interesting.

Usage:
    uv run --group training python -m scripts.labeling.sample_look_noerr
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np

from scripts.labeling.config import MODEL_PATH
from scripts.labeling.train_model import NormalizeTransformer, HandcraftedFeatureExtractor  # noqa: F401

LOGHUB_DIR = Path("/Users/vadimsemenov/datasets/loghub")
SAMPLE_LINES = 50_000  # lines to read per dataset
MAX_EXAMPLES = 20      # examples to print per dataset

_ERROR_RE = re.compile(
    r"\b(ERROR|FATAL|CRITICAL|PANIC|FAIL(?:ED|URE)?|EXCEPTION|SEVERE|EMERG(?:ENCY)?)\b",
    re.IGNORECASE,
)
_WARN_RE = re.compile(r"\b(WARN(?:ING)?)\b", re.IGNORECASE)

DATASETS = [
    ("Android", "Android_v1/Android.log"),
    ("Apache", "Apache/Apache.log"),
    ("BGL", "BGL/BGL.log"),
    ("Hadoop", "Hadoop/container_1445175094696_0003_01_000001.log"),
    ("HDFS", "HDFS/HDFS.log"),
    ("HealthApp", "HealthApp/HealthApp.log"),
    ("HPC", "HPC/HPC.log"),
    ("Linux", "Linux/Linux.log"),
    ("Mac", "Mac/Mac.log"),
    ("OpenStack", "OpenStack/openstack_abnormal.log"),
    ("Proxifier", "Proxifier/Proxifier.log"),
    ("SSH", "SSH/SSH.log"),
    ("Thunderbird", "Thunderbird/Thunderbird.log"),
    ("Windows", "Windows/Windows.log"),
    ("Zookeeper", "Zookeeper/Zookeeper.log"),
]


def main() -> None:
    from joblib import load

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    model = load(MODEL_PATH)
    random.seed(42)

    for name, rel in DATASETS:
        path = LOGHUB_DIR / rel
        if not path.exists():
            continue

        # Read first N lines
        lines: list[str] = []
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= SAMPLE_LINES:
                    break
                lines.append(line.rstrip("\n"))

        if not lines:
            continue

        preds = np.array(model.predict(lines))
        look_mask = preds == "LOOK"
        err_mask = np.array([bool(_ERROR_RE.search(l)) for l in lines])
        warn_mask = np.array([bool(_WARN_RE.search(l)) for l in lines])

        # LOOK lines that are NOT error and NOT warn
        interesting_mask = look_mask & ~err_mask & ~warn_mask
        n_interesting = int(interesting_mask.sum())
        n_look = int(look_mask.sum())
        n_err_look = int((look_mask & err_mask).sum())
        n_warn_look = int((look_mask & warn_mask & ~err_mask).sum())

        print(f"\n{'='*100}")
        print(f"  {name} — {len(lines):,} lines sampled")
        print(f"  LOOK: {n_look:,}  (ERROR: {n_err_look:,}, WARN-only: {n_warn_look:,}, other: {n_interesting:,})")
        print(f"{'='*100}")

        if n_interesting == 0:
            print("  (no LOOK lines without ERROR/WARN keywords)")
            continue

        # Get indices of interesting lines and sample
        indices = np.where(interesting_mask)[0]
        sample_idx = sorted(random.sample(list(indices), min(MAX_EXAMPLES, len(indices))))

        for idx in sample_idx:
            line = lines[idx]
            # Truncate long lines
            if len(line) > 200:
                line = line[:200] + "..."
            print(f"  [{idx:>5d}] {line}")


if __name__ == "__main__":
    main()
