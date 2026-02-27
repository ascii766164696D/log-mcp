"""Fine-tune a small transformer (bert-mini) for LOOK/SKIP log line classification.

Same holdout convention as train_model.py: train on 14+ datasets, test on BGL + Thunderbird.

Usage:
    uv run --group transformer python -m scripts.labeling.train_transformer
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from scripts.labeling.config import ALL_LABELED_PATH, MODELS_DIR
from scripts.labeling.train_model import HOLDOUT_DATASETS, load_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "prajjwal1/bert-mini"  # 11.2M params, 4 layers, 256 hidden
MAX_SEQ_LEN = 256
OUTPUT_DIR = MODELS_DIR / "transformer" / "pytorch"
LABEL2ID = {"LOOK": 0, "SKIP": 1}
ID2LABEL = {0: "LOOK", 1: "SKIP"}


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Weighted loss trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Convert back to string labels for sklearn
    label_names = ["LOOK", "SKIP"]
    y_true = [label_names[l] for l in labels]
    y_pred = [label_names[p] for p in preds]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "look_f1": f1_score(y_true, y_pred, pos_label="LOOK"),
        "look_precision": precision_score(y_true, y_pred, pos_label="LOOK", zero_division=0),
        "look_recall": recall_score(y_true, y_pred, pos_label="LOOK", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Device ---
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log(f"Device: {device}")

    # --- Load data ---
    log(f"Loading data from {ALL_LABELED_PATH} ...")
    if not ALL_LABELED_PATH.exists():
        log(f"ERROR: {ALL_LABELED_PATH} not found. Run the labeling pipeline first.")
        sys.exit(1)

    raw_lines, labels, systems = load_data()
    labels_arr = np.array(labels)
    systems_arr = np.array(systems)

    n_look = np.sum(labels_arr == "LOOK")
    n_skip = np.sum(labels_arr == "SKIP")
    log(f"Loaded {len(raw_lines):,} lines: {n_look:,} LOOK, {n_skip:,} SKIP")

    # --- Train/test split ---
    holdout_mask = np.isin(systems_arr, list(HOLDOUT_DATASETS))
    train_mask = ~holdout_mask

    X_train = [raw_lines[i] for i in range(len(raw_lines)) if train_mask[i]]
    y_train = [LABEL2ID[labels[i]] for i in range(len(labels)) if train_mask[i]]
    X_test = [raw_lines[i] for i in range(len(raw_lines)) if holdout_mask[i]]
    y_test = [LABEL2ID[labels[i]] for i in range(len(labels)) if holdout_mask[i]]
    y_test_str = [labels[i] for i in range(len(labels)) if holdout_mask[i]]
    systems_test = [systems[i] for i in range(len(systems)) if holdout_mask[i]]

    log(f"Train: {len(X_train):,} lines")
    log(f"Test:  {len(X_test):,} lines (holdout: {', '.join(sorted(HOLDOUT_DATASETS))})")

    # --- Compute class weights ---
    train_labels_str = [labels[i] for i in range(len(labels)) if train_mask[i]]
    weights = compute_class_weight("balanced", classes=np.array(["LOOK", "SKIP"]), y=np.array(train_labels_str))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    log(f"Class weights: LOOK={weights[0]:.3f}, SKIP={weights[1]:.3f}")

    # --- Tokenizer ---
    log(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    # --- Build HF datasets ---
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    log("Tokenizing...")
    t0 = time.time()
    train_ds = train_ds.map(tokenize, batched=True, batch_size=1000)
    test_ds = test_ds.map(tokenize, batched=True, batch_size=1000)
    log(f"Tokenized in {time.time() - t0:.1f}s")

    # --- Model ---
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training args ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="look_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=False,  # MPS doesn't support fp16
        report_to="none",
        seed=42,
    )

    # --- Train ---
    log("\nStarting training...")
    t0 = time.time()

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    train_time = time.time() - t0
    log(f"\nTraining done in {train_time:.1f}s")

    # --- Evaluate on holdout ---
    log("\n" + "=" * 60)
    log("HOLDOUT EVALUATION (BGL + Thunderbird)")
    log("=" * 60)

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    y_pred_str = [ID2LABEL[p] for p in preds]

    acc = accuracy_score(y_test_str, y_pred_str)
    log(f"\nAccuracy: {acc:.4f}")
    log(f"\n{classification_report(y_test_str, y_pred_str, digits=4)}")

    # Per-dataset breakdown
    systems_test_arr = np.array(systems_test)
    y_test_str_arr = np.array(y_test_str)
    y_pred_str_arr = np.array(y_pred_str)
    per_dataset = {}

    for ds in sorted(set(systems_test)):
        ds_mask = systems_test_arr == ds
        ds_true = y_test_str_arr[ds_mask]
        ds_pred = y_pred_str_arr[ds_mask]
        ds_metrics = {
            "n": int(ds_mask.sum()),
            "accuracy": round(float(accuracy_score(ds_true, ds_pred)), 4),
            "look_precision": round(float(precision_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
            "look_recall": round(float(recall_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
            "look_f1": round(float(f1_score(ds_true, ds_pred, pos_label="LOOK", zero_division=0)), 4),
        }
        per_dataset[ds] = ds_metrics
        log(f"  {ds}: acc={ds_metrics['accuracy']:.4f}  "
            f"LOOK P={ds_metrics['look_precision']:.4f} "
            f"R={ds_metrics['look_recall']:.4f} "
            f"F1={ds_metrics['look_f1']:.4f} "
            f"(n={ds_metrics['n']})")

    # --- Save best model ---
    best_dir = OUTPUT_DIR / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log(f"\nBest model saved to {best_dir}")

    # --- Save eval report ---
    overall_metrics = {
        "accuracy": round(float(acc), 4),
        "look_precision": round(float(precision_score(y_test_str, y_pred_str, pos_label="LOOK", zero_division=0)), 4),
        "look_recall": round(float(recall_score(y_test_str, y_pred_str, pos_label="LOOK", zero_division=0)), 4),
        "look_f1": round(float(f1_score(y_test_str, y_pred_str, pos_label="LOOK", zero_division=0)), 4),
        "skip_precision": round(float(precision_score(y_test_str, y_pred_str, pos_label="SKIP", zero_division=0)), 4),
        "skip_recall": round(float(recall_score(y_test_str, y_pred_str, pos_label="SKIP", zero_division=0)), 4),
        "skip_f1": round(float(f1_score(y_test_str, y_pred_str, pos_label="SKIP", zero_division=0)), 4),
    }

    eval_report = {
        "model": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "train_time_s": round(train_time, 1),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "holdout_datasets": sorted(HOLDOUT_DATASETS),
        "class_weights": {"LOOK": round(weights[0], 3), "SKIP": round(weights[1], 3)},
        "overall": overall_metrics,
        "per_dataset": per_dataset,
    }

    report_path = MODELS_DIR / "eval_transformer.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    log(f"Eval report saved to {report_path}")

    # --- Compare with TF-IDF baseline ---
    tfidf_report_path = MODELS_DIR / "eval_report.json"
    if tfidf_report_path.exists():
        with open(tfidf_report_path) as f:
            tfidf_report = json.load(f)
        log("\n" + "=" * 60)
        log("COMPARISON: TF-IDF vs Transformer")
        log("=" * 60)
        tfidf_m = tfidf_report["overall"]
        log(f"{'Metric':<20s} {'TF-IDF':>10s} {'Transformer':>12s} {'Delta':>10s}")
        log("-" * 55)
        for metric in ["look_precision", "look_recall", "look_f1", "accuracy"]:
            t_val = tfidf_m[metric]
            x_val = overall_metrics[metric]
            delta = x_val - t_val
            sign = "+" if delta >= 0 else ""
            log(f"{metric:<20s} {t_val:>10.4f} {x_val:>12.4f} {sign}{delta:>9.4f}")


if __name__ == "__main__":
    main()
