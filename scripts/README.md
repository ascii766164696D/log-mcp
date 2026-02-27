# Scripts

## generate_test_logs.py

Generates two synthetic log files (`data/test.log`, `data/test2.log`, ~1500 lines each) simulating two servers of a web application. Includes shared patterns at different frequencies, unique error patterns per server, realistic IDs, and multi-line stack traces. Uses a fixed seed for reproducibility.

```bash
uv run python -m scripts.generate_test_logs
```

## labeling/

ML pipeline for training the LOOK/SKIP classifier. See [labeling/RETRAINING.md](labeling/RETRAINING.md) for the full retraining guide.

### Data acquisition

| Script | Purpose |
|---|---|
| `download_loghub.py` | Downloads 2K-line samples from loghub (22 datasets). Idempotent. |
| `generate_error_logs.py` | Generates 6 synthetic error-heavy log files for underrepresented patterns (OOM, kernel panics, deadlocks, 5xx, gRPC failures, brute-force). |

### Claude labeling (Batch API)

Each 2K-line file is split into 50-line windows and sent to Claude Haiku for LOOK/SKIP labeling via the Batch API (~50% cost vs synchronous).

| Script | Purpose |
|---|---|
| `config.py` | Paths, dataset registry, labeling prompts, model name. |
| `prepare_batches.py` | Builds Batch API request JSONL from loghub samples. `--dry-test` tests one window first. |
| `submit_batch.py` | Submits requests to the Batch API. Saves batch ID to manifest. |
| `collect_results.py` | Polls until complete, streams results to JSONL. |
| `parse_labels.py` | Maps window-relative labels back to absolute line numbers. Writes `all_labeled.jsonl`. |
| `validate_labels.py` | Checks labels against BGL/Thunderbird ground truth (precision, recall, F1). |
| `label_new.py` | Incremental wrapper — only labels datasets missing from `all_labeled.jsonl`. |
| `run_pipeline.py` | Orchestrates: download → prepare → submit → collect → parse → validate. `--skip-batch` re-parses without re-submitting. |

### Model training & export

| Script | Purpose |
|---|---|
| `train_model.py` | Trains TF-IDF + logistic regression (word n-grams, char n-grams, 17 handcrafted features). GroupKFold CV, tests on BGL + Thunderbird holdout. |
| `export_model.py` | Exports sklearn model to JSON (6.7 MB) for the Rust classifier. |
| `train_transformer.py` | Fine-tunes `prajjwal1/bert-mini` (11M params) on the same labels. Optional second-stage classifier. |
| `export_transformer.py` | Exports BERT checkpoint to safetensors for Rust candle runtime. |

### Evaluation

| Script | Purpose |
|---|---|
| `eval_all_loghub.py` | Runs TF-IDF model on full loghub datasets (parallel). Reports per-dataset LOOK/SKIP stats. |
| `eval_full_loghub.py` | Full eval with F1 on BGL ground truth + LOOK distribution for all datasets. |
| `eval_rust_loghub.py` | Benchmarks the compiled Rust classifier on all loghub datasets. |
| `sample_look_noerr.py` | Prints LOOK lines that have no ERROR/WARN keywords — shows what else the model finds interesting. |

### Quick start

**Retrain after adding new log samples:**

```bash
# 1. Add 2K-line samples to data/loghub/MyApp_2k.log and register in config.DATASETS
# 2. Label only new datasets (incremental)
uv run python -m scripts.labeling.label_new
# 3. Train and export
uv run --group training python -m scripts.labeling.train_model
uv run --group training python -m scripts.labeling.export_model
# 4. Rebuild Rust classifier
uv pip install -e rust/classifier
```

**Full pipeline from scratch:**

```bash
uv run python -m scripts.labeling.run_pipeline
uv run --group training python -m scripts.labeling.train_model
uv run --group training python -m scripts.labeling.export_model
```

### Data layout

```
data/
  loghub/                          # 2K-line samples (22 datasets)
  labels/all_labeled.jsonl         # Combined LOOK/SKIP labels
  batches/                         # Batch API artifacts
  models/
    look_skip_model.json           # TF-IDF model for Rust
    look_skip_model.joblib         # sklearn model (Python)
    transformer/export/            # BERT-mini for Rust candle
```
