# look-skip-classifier

Rust + PyO3 native extension that classifies log lines as LOOK (interesting) or SKIP (routine). Used by log-mcp's `classify_lines` tool.

## How it works

Two classifier backends, exposed as Python classes via PyO3:

### `LookSkipClassifier` — TF-IDF + logistic regression

A port of the scikit-learn pipeline trained on 17 loghub datasets (345M lines). Processes ~1.3M lines/sec.

Pipeline per line:
1. **Normalize** — regex-based replacement of UUIDs, hex, IPs, numbers, quoted strings, IDs
2. **Word TF-IDF** — n-gram vectorization on normalized text (matches sklearn `TfidfVectorizer(analyzer="word")`)
3. **Char TF-IDF** — character n-gram vectorization with word boundaries on raw text (matches sklearn `analyzer="char_wb"`)
4. **Handcrafted features** — 17 features: line length, uppercase ratio, has log level, digit ratio, has stack trace, structural punctuation ratio, negative language, word count, allcaps words, leading whitespace, has path, hex count, repeated punctuation, key-value pairs, unique word ratio, emphasis punctuation
5. **Dot product** — sparse dot product against sklearn's `coef_` vector + intercept, sigmoid to get P(LOOK)

File classification uses pipelined I/O (background reader thread filling 128 MB chunks) with rayon parallel classification.

### `TransformerClassifier` — BERT-mini (optional)

Fine-tuned `prajjwal1/bert-mini` (11M params) loaded via candle. Runs on Metal GPU when available, falls back to CPU. ~2K lines/sec on M-series Macs. Uses dynamic padding (pad to longest in batch, not max_seq_len) for efficiency.

Gated behind the `transformer` feature flag.

## Modules

| File | Purpose |
|------|---------|
| `lib.rs` | PyO3 module: `LookSkipClassifier` and `TransformerClassifier` Python classes |
| `model.rs` | Model JSON deserialization, regex compilation, `HashLookup` (open-addressing hash table for fast n-gram matching) |
| `classifier.rs` | `classify_line`, `classify_file` (pipelined I/O + rayon), `classify_file_with_keywords` (ERROR/WARN counting) |
| `tfidf.rs` | Word and char_wb TF-IDF vectorization with thread-local buffers, L2 normalization |
| `features.rs` | 17 handcrafted feature extraction (single-pass byte scan where possible) |
| `normalize.rs` | Regex-based text normalization, custom ID replacement (avoids fancy_regex lookahead) |
| `transformer.rs` | BERT-mini inference via candle: tokenization, batched forward pass, softmax |

## Build

Requires Rust toolchain and maturin:

```bash
# Development build (into current venv)
pip install maturin
maturin develop --release

# With Metal GPU acceleration for transformer
maturin develop --release --features metal
```

Feature flags:

| Flag | Effect |
|------|--------|
| `transformer` (default) | Enables `TransformerClassifier` via candle |
| `metal` | Metal GPU acceleration for candle (Apple Silicon) |
| `accelerate` | Accelerate framework for candle (Apple Silicon CPU BLAS) |

## Usage from Python

```python
from look_skip_classifier import LookSkipClassifier

clf = LookSkipClassifier("path/to/model.json")

# Single line
label, prob = clf.classify_line("ERROR: connection refused", threshold=0.5)
# ("LOOK", 0.92)

# Whole file (pipelined I/O + parallel classification)
result = clf.classify_file("app.log", threshold=0.5, max_lines=0, max_look_lines=200)
# {"total_lines": 705000, "look_count": 3200, "skip_count": 701800,
#  "processing_time_s": 0.54, "lines_per_second": 1305555.0,
#  "look_lines": [(42, 0.95, "ERROR: connection refused"), ...]}
```

## Model format

`LookSkipClassifier` loads a single JSON file containing:

```
{
  "word_tfidf":    { vocab, idf, ngram_range, sublinear_tf },
  "char_tfidf":    { vocab, idf, ngram_range, sublinear_tf },
  "normalize_patterns": [ { pattern, replacement, case_insensitive }, ... ],
  "handcrafted":   { level_pattern, stack_trace_pattern, ... },
  "classifier":    { coef, intercept, classes, n_word_features, n_char_features, n_handcrafted_features }
}
```

`TransformerClassifier` loads a directory with `config.json`, `model.safetensors`, and `tokenizer.json` (HuggingFace format).

## Tests

```bash
cargo test --lib
```

(`--lib` skips the `cdylib` target which requires Python symbols to link.)
