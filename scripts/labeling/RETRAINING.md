# Retraining the LOOK/SKIP Classifier

The classifier ships pre-trained on 22 datasets (~40K labeled lines), but you can retrain it on your own logs to improve accuracy for your specific log formats.

## How labeling works

Claude labels log lines via the Anthropic Batch API. Each 2K-line log file is split into windows of 50 consecutive lines. Claude sees each window with this prompt:

```
System: {dataset_name}
Lines 1-50 (of 2000):

1: 081231 13:55:36 compute-1-13 kernel: [8796699.725147] Parity Error ...
2: 081231 13:55:36 compute-1-13 kernel: [8796699.725148] CPU: Physical Processor ID: 0
...
50: 081231 13:55:43 compute-1-13 kernel: HANDLING MCE MEMORY ERROR
```

Claude returns a JSON array classifying each line:
```json
[
  {"line": 1, "label": "LOOK", "reason": "Parity error detected"},
  {"line": 2, "label": "SKIP", "reason": "Normal CPU info"},
  {"line": 50, "label": "LOOK", "reason": "Memory error (MCE)"}
]
```

The system prompt tells Claude to lean toward LOOK when in doubt — false positives are cheaper than missed anomalies. You can customize the prompt in `config.py`.

**Cost**: ~$0.10-0.50 per 2K-line file using Claude Haiku via the Batch API (50% discount vs synchronous).

## How training works

### TF-IDF + Logistic Regression

The model combines three feature sources into a 115K-dimensional feature vector:

**Word TF-IDF (35K features)**: Log lines are normalized (UUIDs, IPs, hex, numbers replaced with placeholders), then vectorized with word n-grams (1-3). This lets the model match patterns like `"connection <PLACEHOLDER> refused"` regardless of the specific IP.

**Char TF-IDF (80K features)**: Raw character n-grams (3-6 chars). Catches subword patterns like `"rror"`, `"fail"`, `"imeou"` that appear across different error messages.

**Handcrafted features (17 features)**:
| Feature | What it captures |
|---|---|
| `has_level` | Contains ERROR, FATAL, CRITICAL, WARN, EXCEPTION, FAIL |
| `has_stack` | Looks like a stack trace (`Traceback`, `at com.foo`, `Caused by:`) |
| `has_negative` | Negative keywords: unable, cannot, timeout, denied, refused, panic, OOM, etc. |
| `n_negative` | Count of negative keyword matches |
| `has_path` | Contains file paths (`/var/log/...`) |
| `n_hex` | Count of hex values (memory addresses, error codes) |
| `has_repeated_punct` | Separator lines (`===`, `---`) |
| `n_kv` | Count of key=value pairs |
| `uppercase_ratio` | Fraction of uppercase characters |
| `digit_ratio` | Fraction of digit characters |
| `struct_punct` | Density of structural punctuation (`:;=()[]{}`) |
| `word_count` | Number of words |
| `n_allcaps` | Count of ALL-CAPS words (3+ chars) |
| `unique_ratio` | Word diversity (unique / total) |
| `leading_ws` | Leading whitespace length (stack trace continuation) |
| `n_emphasis` | Count of `!` and `?` |
| `length` | Raw character count |

Training uses `GroupKFold` cross-validation with dataset as the group key, so the model is never evaluated on data from the same log source it trained on. Grid search tunes regularization strength and feature counts.

The trained sklearn pipeline is exported to a 6.7 MB JSON file that the Rust classifier loads at startup.

### BERT-mini (optional)

Fine-tunes `prajjwal1/bert-mini` (4 layers, 256 hidden, 11M params) on the same labels. Uses weighted cross-entropy to handle class imbalance, early stopping on LOOK F1 (patience=3), and trains for up to 10 epochs.

Exported as safetensors for Rust candle inference with Metal GPU acceleration. The Rust classifier runs BERT on the TF-IDF LOOK lines only, so it only processes 5-30% of the file.

## Quick start: add your logs and retrain

**Prerequisites:**
```bash
# Anthropic API key (for Claude labeling)
export ANTHROPIC_API_KEY=sk-ant-...

# Install dependencies
uv sync --group labeling --group training
```

**Step 1: Add your log files**

Place 2K-line samples in `data/loghub/`:

```bash
# Take a sample from your production logs
head -2000 /var/log/myapp/app.log > data/loghub/MyApp_2k.log
```

Then register the dataset in `scripts/labeling/config.py`:
```python
DATASETS: list[tuple[str, bool]] = [
    ...
    ("MyApp", False),  # False = no ground truth CSV
]
```

**Step 2: Label with Claude**

```bash
# Label only new (unlabeled) datasets
uv run python -m scripts.labeling.label_new

# Or dry-run first to see what will be labeled
uv run python -m scripts.labeling.label_new --dry-run
```

`label_new` detects which datasets already have labels in `all_labeled.jsonl` and only submits new ones. It appends results — existing labels are preserved.

**Step 3: Train the TF-IDF model**

```bash
uv run --group training python -m scripts.labeling.train_model
uv run --group training python -m scripts.labeling.export_model
```

`train_model` runs grid search (32 hyperparameter combinations x 5 CV folds). `export_model` converts the sklearn pipeline to JSON and validates prediction parity on test lines.

**Step 4: (Optional) Train the BERT model**

```bash
uv run --group transformer python -m scripts.labeling.train_transformer
uv run --group transformer python -m scripts.labeling.export_transformer
```

`export_transformer` copies model.safetensors, config.json, tokenizer.json to `data/models/transformer/export/` and fixes LayerNorm key names for Candle compatibility.

**Step 5: Rebuild the Rust classifier**

```bash
uv pip install -e rust/classifier
```

## Full pipeline from scratch

To redo everything (download loghub samples, label all, train, export):

```bash
uv run python -m scripts.labeling.run_pipeline
uv run --group training python -m scripts.labeling.train_model
uv run --group training python -m scripts.labeling.export_model
```

`run_pipeline` runs: download loghub → prepare batch requests → submit to Claude → poll until done → parse labels → validate against ground truth (BGL, Thunderbird).

## Customization

**Change the labeling prompt**: Edit `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` in `config.py`. The system prompt controls what Claude considers LOOK vs SKIP. If your logs have domain-specific anomalies (e.g., Zookeeper state transitions), add them to the LOOK criteria.

**Change the labeling model**: Edit `MODEL` in `config.py`. Default is `claude-haiku-4-5` (fast and cheap). Using a larger model costs more but may label more accurately.

**Change the window size**: Edit `WINDOW_SIZE` in `config.py`. Larger windows give Claude more context but cost more per request. Default is 50 lines.

**Add handcrafted features**: Edit `HandcraftedFeatureExtractor.transform()` in `train_model.py` to add domain-specific patterns, then retrain.

## Scripts reference

| Script | What it does | Command |
|---|---|---|
| `download_loghub` | Download 2K-line loghub samples | `uv run python -m scripts.labeling.download_loghub` |
| `generate_error_logs` | Generate synthetic error-heavy logs | `uv run python -m scripts.labeling.generate_error_logs` |
| `prepare_batches` | Create Batch API requests (50-line windows) | `uv run python -m scripts.labeling.prepare_batches` |
| `submit_batch` | Submit to Anthropic Batch API | `uv run python -m scripts.labeling.submit_batch` |
| `collect_results` | Poll and download batch results | `uv run python -m scripts.labeling.collect_results` |
| `parse_labels` | Extract per-line LOOK/SKIP labels | `uv run python -m scripts.labeling.parse_labels` |
| `validate_labels` | Compare against BGL/Thunderbird ground truth | `uv run python -m scripts.labeling.validate_labels` |
| `label_new` | Label only new/unlabeled datasets | `uv run python -m scripts.labeling.label_new` |
| `train_model` | Train TF-IDF + LogReg classifier | `uv run --group training python -m scripts.labeling.train_model` |
| `export_model` | Export to JSON for Rust | `uv run --group training python -m scripts.labeling.export_model` |
| `train_transformer` | Fine-tune BERT-mini | `uv run --group transformer python -m scripts.labeling.train_transformer` |
| `export_transformer` | Export BERT for Rust candle | `uv run --group transformer python -m scripts.labeling.export_transformer` |
| `run_pipeline` | Run full pipeline end-to-end | `uv run python -m scripts.labeling.run_pipeline` |
| `eval_rust_loghub` | Benchmark Rust classifier on full loghub | `python scripts/labeling/eval_rust_loghub.py` |

## Data layout

```
data/
  loghub/           # 2K-line log samples (22 datasets)
  labels/           # Claude LOOK/SKIP labels per dataset
    all_labeled.jsonl  # Combined training set (~40K lines)
  batches/          # Anthropic Batch API artifacts
    batch_requests.jsonl   # All prepared requests
    batch_results.jsonl    # All collected responses
    manifest.json          # Batch ID and status
  models/
    look_skip_model.joblib            # sklearn pipeline (for Python)
    look_skip_model.json              # TF-IDF model for Rust (6.7 MB)
    transformer/
      pytorch/best/                   # Fine-tuned BERT checkpoint
      export/                         # BERT model for Rust candle (43 MB)
        config.json
        model.safetensors
        tokenizer.json
        reference_outputs.json        # Test predictions for parity checking
    eval_report.json                  # TF-IDF cross-validation results
    eval_rust_loghub.json             # Rust classifier benchmark (full loghub)
    eval_transformer.json             # BERT evaluation results
```
