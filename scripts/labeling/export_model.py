"""Export the trained sklearn LOOK/SKIP model to a portable JSON format.

Extracts TF-IDF vocabularies, IDF weights, logistic regression coefficients,
normalization patterns, and handcrafted feature regexes from the joblib model
into a JSON file that the Rust classifier can load.

Usage:
    uv run --group training python -m scripts.labeling.export_model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.labeling.config import MODEL_PATH
from scripts.labeling.train_model import NormalizeTransformer, HandcraftedFeatureExtractor  # noqa: F401


EXPORT_PATH = MODEL_PATH.parent / "look_skip_model.json"


def main() -> None:
    from joblib import load

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} ...")
    model = load(MODEL_PATH)

    # Navigate the pipeline structure:
    # Pipeline([("features", FeatureUnion([...])), ("clf", LogisticRegression(...))])
    features_union = model.named_steps["features"]
    clf = model.named_steps["clf"]

    # Extract the 3 feature transformers from the FeatureUnion
    transformers = dict(features_union.transformer_list)

    # 1. Word TF-IDF (on normalized text)
    word_pipeline = transformers["tfidf_normalized"]  # Pipeline([normalize, tfidf])
    word_tfidf = word_pipeline.named_steps["tfidf"]

    word_vocab = {k: int(v) for k, v in word_tfidf.vocabulary_.items()}
    word_idf = word_tfidf.idf_          # ndarray
    print(f"Word TF-IDF: {len(word_vocab)} features, ngram_range={word_tfidf.ngram_range}, "
          f"sublinear_tf={word_tfidf.sublinear_tf}")

    # 2. Char TF-IDF (on raw text)
    char_tfidf = transformers["tfidf_char"]

    char_vocab = {k: int(v) for k, v in char_tfidf.vocabulary_.items()}
    char_idf = char_tfidf.idf_
    print(f"Char TF-IDF: {len(char_vocab)} features, ngram_range={char_tfidf.ngram_range}")

    # 3. Handcrafted features (17 features, no learned params)
    n_handcrafted = 17
    print(f"Handcrafted: {n_handcrafted} features")

    # 4. Classifier
    coef = clf.coef_[0]          # shape (n_features,) for binary LR
    intercept = clf.intercept_[0]
    classes = list(clf.classes_)
    print(f"Classifier: coef shape={coef.shape}, intercept={intercept:.6f}, classes={classes}")
    print(f"Total features: {len(word_vocab)} + {len(char_vocab)} + {n_handcrafted} = {len(word_vocab) + len(char_vocab) + n_handcrafted}")
    assert coef.shape[0] == len(word_vocab) + len(char_vocab) + n_handcrafted, \
        f"Coefficient dimension mismatch: {coef.shape[0]} != {len(word_vocab) + len(char_vocab) + n_handcrafted}"

    # Build export dict
    export = {
        "word_tfidf": {
            "vocab": word_vocab,
            "idf": word_idf.tolist(),
            "ngram_range": list(word_tfidf.ngram_range),
            "sublinear_tf": bool(word_tfidf.sublinear_tf),
        },
        "char_tfidf": {
            "vocab": char_vocab,
            "idf": char_idf.tolist(),
            "ngram_range": list(char_tfidf.ngram_range),
        },
        "normalize_patterns": [
            {"pattern": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "replacement": "<UUID>", "case_insensitive": True},
            {"pattern": r"\b[0-9a-f]{12,}\b", "replacement": "<HEX>", "case_insensitive": True},
            {"pattern": r"0x[0-9a-fA-F]+", "replacement": "<HEX>", "case_insensitive": False},
            {"pattern": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "replacement": "<IP>", "case_insensitive": False},
            {"pattern": r"\b[a-z]+_(?=[a-z]*\d)[a-z0-9]{4,}\b", "replacement": "<ID>", "case_insensitive": True},
            {"pattern": r"\b\d+", "replacement": "<N>", "case_insensitive": False},
            {"pattern": r'"[^"]*"', "replacement": '"<STR>"', "case_insensitive": False},
            {"pattern": r"'[^']*'", "replacement": "'<STR>'", "case_insensitive": False},
        ],
        "handcrafted": {
            "level_pattern": r"ERROR|FATAL|CRITICAL|WARN|EXCEPTION|FAIL",
            "stack_trace_pattern": r'Traceback|^\s+at\s|Caused by:|^\s+File\s"',
            "negative_pattern": (
                r"\b(?:unable|cannot|couldn't|can't|not found|no such|denied|refused"
                r"|rejected|timeout|timed out|unreachable|unavailable|unknown|invalid"
                r"|illegal|unexpected|unsupported|mismatch|overflow|underflow"
                r"|corrupt|missing|lost|kill|abort|panic|segfault|oom|no answer"
                r"|not responding|sleeping for)\b"
            ),
            "path_pattern": r"/[\w./-]{3,}",
            "hex_pattern": r"\b(?:0x[0-9a-fA-F]+|[0-9a-f]{8,})\b",
            "repeated_punct_pattern": r"(.)\1{2,}",
            "key_value_pattern": r"\b\w+=\S+",
        },
        "classifier": {
            "coef": coef.tolist(),
            "intercept": float(intercept),
            "classes": classes,
            "n_word_features": len(word_vocab),
            "n_char_features": len(char_vocab),
            "n_handcrafted_features": n_handcrafted,
        },
    }

    # Write JSON
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f)
    size_mb = EXPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nExported to {EXPORT_PATH} ({size_mb:.1f} MB)")

    # Quick validation: load back and verify shapes
    print("\nValidating export ...")
    with open(EXPORT_PATH) as f:
        loaded = json.load(f)

    assert len(loaded["word_tfidf"]["vocab"]) == len(word_vocab)
    assert len(loaded["word_tfidf"]["idf"]) == len(word_idf)
    assert len(loaded["char_tfidf"]["vocab"]) == len(char_vocab)
    assert len(loaded["char_tfidf"]["idf"]) == len(char_idf)
    assert len(loaded["classifier"]["coef"]) == coef.shape[0]
    assert abs(loaded["classifier"]["intercept"] - intercept) < 1e-10

    # Verify prediction parity on a few sample lines
    samples = [
        "2024-01-15 10:00:01 ERROR Connection timeout to database server 192.168.1.5",
        "2024-01-15 10:00:02 INFO Health check passed, all systems nominal",
        "Traceback (most recent call last):",
        "  at com.example.Main.run(Main.java:42)",
        "Jan 15 10:00:03 server01 sshd[12345]: Invalid user admin from 10.0.0.1",
    ]
    sklearn_probs = clf.predict_proba(
        model.named_steps["features"].transform(samples)
    )
    look_idx = list(clf.classes_).index("LOOK")
    print(f"\nSample predictions (sklearn P(LOOK)):")
    for line, probs in zip(samples, sklearn_probs):
        short = line[:60] + "..." if len(line) > 60 else line
        print(f"  {probs[look_idx]:.6f}  {short}")

    print("\nExport complete. Rust classifier should produce identical probabilities.")


if __name__ == "__main__":
    main()
