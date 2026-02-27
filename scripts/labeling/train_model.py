"""Train a TF-IDF + LogisticRegression classifier for LOOK/SKIP log line filtering.

Leave-dataset-out evaluation: trains on 14 datasets, tests on BGL + Thunderbird
(the only ones with ground truth).

Usage:
    uv run --group training python -m scripts.labeling.train_model
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from log_mcp.normalize import normalize_extended
from scripts.labeling.config import (
    ALL_LABELED_PATH,
    EVAL_REPORT_PATH,
    MODEL_PATH,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Holdout datasets (have ground truth — used for final evaluation)
# ---------------------------------------------------------------------------
HOLDOUT_DATASETS = {"BGL", "Thunderbird"}

# ---------------------------------------------------------------------------
# Regex for handcrafted features
# ---------------------------------------------------------------------------
_LEVEL_RE = re.compile(
    r"ERROR|FATAL|CRITICAL|WARN|EXCEPTION|FAIL", re.IGNORECASE
)
_STACK_TRACE_RE = re.compile(
    r"Traceback|^\s+at\s|Caused by:|^\s+File\s\"", re.IGNORECASE
)
_NEGATIVE_RE = re.compile(
    r"\b(?:unable|cannot|couldn't|can't|not found|no such|denied|refused"
    r"|rejected|timeout|timed out|unreachable|unavailable|unknown|invalid"
    r"|illegal|unexpected|unsupported|mismatch|overflow|underflow"
    r"|corrupt|missing|lost|kill|abort|panic|segfault|oom|no answer"
    r"|not responding|sleeping for)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"/[\w./-]{3,}")
_HEX_RE = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-f]{8,})\b")
_REPEATED_PUNCT_RE = re.compile(r"(.)\1{2,}")  # 3+ repeated chars
_KEY_VALUE_RE = re.compile(r"\b\w+=\S+")


# ---------------------------------------------------------------------------
# Custom sklearn transformers
# ---------------------------------------------------------------------------
class NormalizeTransformer(BaseEstimator, TransformerMixin):
    """Applies normalize_extended() to each input string.

    Used as a preprocessing step before TF-IDF so that UUIDs, hex, IPs,
    and numbers are replaced with placeholders.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_extended(line) for line in X]


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts numeric features from raw log lines, returns sparse matrix.

    Features 0-5 are the original 6 features (kept for backward compat).
    Features 6+ are new additions.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for line in X:
            length = len(line)
            denom = max(length, 1)
            n_upper = sum(1 for c in line if c.isupper())
            n_digits = sum(1 for c in line if c.isdigit())
            uppercase_ratio = n_upper / denom
            digit_ratio = n_digits / denom
            has_level = 1.0 if _LEVEL_RE.search(line) else 0.0
            has_stack = 1.0 if _STACK_TRACE_RE.search(line) else 0.0
            struct_punct = sum(1 for c in line if c in ":;=()[]{}") / denom

            # --- New features ---
            # Negative/failure language (captures anomalies without standard level keywords)
            has_negative = 1.0 if _NEGATIVE_RE.search(line) else 0.0
            n_negative = len(_NEGATIVE_RE.findall(line))

            # Word count
            words = line.split()
            word_count = len(words)

            # All-caps words (e.g., TIMEOUT, DENIED, KERNEL, SEVERE)
            n_allcaps = sum(1 for w in words if len(w) >= 3 and w.isupper() and w.isalpha())

            # Leading whitespace (continuation lines, stack traces, indented output)
            leading_ws = len(line) - len(line.lstrip()) if line else 0

            # Has file path
            has_path = 1.0 if _PATH_RE.search(line) else 0.0

            # Has hex values (register dumps like "iar 00106570 dear 0245a")
            n_hex = len(_HEX_RE.findall(line))

            # Has repeated punctuation (=== or --- separators)
            has_repeated_punct = 1.0 if _REPEATED_PUNCT_RE.search(line) else 0.0

            # Has key=value pairs (structured data)
            n_kv = len(_KEY_VALUE_RE.findall(line))

            # Unique word ratio (error messages tend to be more diverse)
            unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)

            # Exclamation/question marks
            n_emphasis = sum(1 for c in line if c in "!?")

            rows.append([
                # Original 6 features (indices 0-5)
                length,
                uppercase_ratio,
                has_level,
                digit_ratio,
                has_stack,
                struct_punct,
                # New features (indices 6+)
                has_negative,       # 6
                n_negative,         # 7
                word_count,         # 8
                n_allcaps,          # 9
                leading_ws,         # 10
                has_path,           # 11
                n_hex,              # 12
                has_repeated_punct, # 13
                n_kv,               # 14
                unique_ratio,       # 15
                n_emphasis,         # 16
            ])
        return csr_matrix(np.array(rows, dtype=np.float64))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path=ALL_LABELED_PATH):
    """Load labeled JSONL, return (raw_lines, labels, systems)."""
    raw_lines = []
    labels = []
    systems = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            raw_lines.append(entry["raw_line"])
            labels.append(entry["label"])
            systems.append(entry["system"])
    return raw_lines, labels, systems


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
def build_pipeline():
    """Build the full sklearn Pipeline with FeatureUnion + LogisticRegression."""
    features = FeatureUnion([
        ("tfidf_normalized", Pipeline([
            ("normalize", NormalizeTransformer()),
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=50000,
                sublinear_tf=True,
            )),
        ])),
        ("tfidf_char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            max_features=80000,
        )),
        ("handcrafted", HandcraftedFeatureExtractor()),
    ])

    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Load data ---
    print(f"Loading data from {ALL_LABELED_PATH} ...")
    if not ALL_LABELED_PATH.exists():
        print(f"ERROR: {ALL_LABELED_PATH} not found. Run the labeling pipeline first.")
        sys.exit(1)

    raw_lines, labels, systems = load_data()
    labels_arr = np.array(labels)
    systems_arr = np.array(systems)

    n_look = np.sum(labels_arr == "LOOK")
    n_skip = np.sum(labels_arr == "SKIP")
    print(f"Loaded {len(raw_lines):,} lines: {n_look:,} LOOK ({n_look/len(raw_lines)*100:.1f}%), "
          f"{n_skip:,} SKIP ({n_skip/len(raw_lines)*100:.1f}%)")

    # --- Per-dataset breakdown ---
    print(f"\n{'Dataset':20s} {'Total':>7s} {'LOOK':>7s} {'SKIP':>7s} {'Split':>7s}")
    print("-" * 50)
    unique_systems = sorted(set(systems))
    for ds in unique_systems:
        mask = systems_arr == ds
        n = mask.sum()
        nl = np.sum(labels_arr[mask] == "LOOK")
        ns = n - nl
        split = "TEST" if ds in HOLDOUT_DATASETS else "TRAIN"
        print(f"{ds:20s} {n:7d} {nl:7d} {ns:7d} {split:>7s}")

    # --- Train/test split (leave-dataset-out) ---
    holdout_mask = np.isin(systems_arr, list(HOLDOUT_DATASETS))
    train_mask = ~holdout_mask

    X_train = [raw_lines[i] for i in range(len(raw_lines)) if train_mask[i]]
    y_train = labels_arr[train_mask]
    groups_train = systems_arr[train_mask]

    X_test = [raw_lines[i] for i in range(len(raw_lines)) if holdout_mask[i]]
    y_test = labels_arr[holdout_mask]
    systems_test = systems_arr[holdout_mask]

    print(f"\nTrain: {len(X_train):,} lines ({len(set(groups_train))} datasets)")
    print(f"Test:  {len(X_test):,} lines ({len(set(systems_test))} datasets: "
          f"{', '.join(sorted(set(systems_test)))})")

    # --- Grid search ---
    pipeline = build_pipeline()

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": ["balanced", {"LOOK": 2, "SKIP": 1}],
        "features__tfidf_normalized__tfidf__max_features": [30000, 50000],
        "features__tfidf_char__max_features": [50000, 80000],
    }

    scorer = lambda est, X, y: f1_score(y, est.predict(X), pos_label="LOOK")

    group_kfold = GroupKFold(n_splits=5)

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    n_fits = n_combos * group_kfold.n_splits
    print(f"\nStarting GridSearchCV ({n_combos} combinations × {group_kfold.n_splits} folds = {n_fits} fits) ...")
    t0 = time.time()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scorer,
        cv=group_kfold,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=groups_train)

    elapsed = time.time() - t0
    print(f"Grid search done in {elapsed:.1f}s")
    print(f"Best LOOK F1 (CV): {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    # --- Evaluate on holdout ---
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION (BGL + Thunderbird)")
    print("=" * 60)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, digits=4)}")

    # Per-dataset breakdown
    per_dataset = {}
    for ds in sorted(set(systems_test)):
        ds_mask = systems_test == ds
        ds_pred = y_pred[ds_mask]
        ds_true = y_test[ds_mask]
        ds_metrics = {
            "n": int(ds_mask.sum()),
            "accuracy": round(float(accuracy_score(ds_true, ds_pred)), 4),
            "look_precision": round(float(precision_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
            "look_recall": round(float(recall_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
            "look_f1": round(float(f1_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
        }
        per_dataset[ds] = ds_metrics
        print(f"  {ds}: acc={ds_metrics['accuracy']:.4f}  "
              f"LOOK P={ds_metrics['look_precision']:.4f} "
              f"R={ds_metrics['look_recall']:.4f} "
              f"F1={ds_metrics['look_f1']:.4f} "
              f"(n={ds_metrics['n']})")

    # --- Save model + eval report ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from joblib import dump
    dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    overall_metrics = {
        "accuracy": round(float(acc), 4),
        "look_precision": round(float(precision_score(y_test, y_pred, pos_label="LOOK", zero_division=0)), 4),
        "look_recall": round(float(recall_score(y_test, y_pred, pos_label="LOOK", zero_division=0)), 4),
        "look_f1": round(float(f1_score(y_test, y_pred, pos_label="LOOK", zero_division=0)), 4),
        "skip_precision": round(float(precision_score(y_test, y_pred, pos_label="SKIP", zero_division=0)), 4),
        "skip_recall": round(float(recall_score(y_test, y_pred, pos_label="SKIP", zero_division=0)), 4),
        "skip_f1": round(float(f1_score(y_test, y_pred, pos_label="SKIP", zero_division=0)), 4),
    }

    eval_report = {
        "best_params": {k: v if not isinstance(v, dict) else v for k, v in grid.best_params_.items()},
        "best_cv_look_f1": round(float(grid.best_score_), 4),
        "grid_search_time_s": round(elapsed, 1),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "holdout_datasets": sorted(HOLDOUT_DATASETS),
        "overall": overall_metrics,
        "per_dataset": per_dataset,
    }

    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"Eval report saved to {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
