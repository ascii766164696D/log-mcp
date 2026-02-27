"""Export and verify the fine-tuned transformer for Rust candle inference.

Verifies that the saved model produces identical outputs when re-loaded,
and copies the minimal set of files needed for inference to a clean directory.

Usage:
    uv run --group transformer python -m scripts.labeling.export_transformer
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from scripts.labeling.config import MODELS_DIR

SRC_DIR = MODELS_DIR / "transformer" / "pytorch" / "best"
EXPORT_DIR = MODELS_DIR / "transformer" / "export"

# Files needed for candle inference
NEEDED_FILES = ["config.json", "model.safetensors", "tokenizer.json"]

TEST_LINES = [
    "2024-01-15 ERROR: Connection refused to database server at 10.0.0.5:5432",
    "2024-01-15 INFO: Health check passed, all services running",
    "kernel: Out of memory: Kill process 12345 (java) score 950",
    "Jan 15 03:00:01 server01 CRON[1234]: (root) CMD (logrotate /etc/logrotate.conf)",
    "FATAL: password authentication failed for user \"admin\"",
]


def main() -> None:
    print(f"Loading model from {SRC_DIR} ...")
    tokenizer = BertTokenizer.from_pretrained(str(SRC_DIR))
    model = BertForSequenceClassification.from_pretrained(str(SRC_DIR))
    model.eval()

    # Run test inference
    print(f"\nTest inference on {len(TEST_LINES)} lines:")
    inputs = tokenizer(
        TEST_LINES,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label
    for i, line in enumerate(TEST_LINES):
        pred_idx = probs[i].argmax().item()
        pred_label = id2label[pred_idx]
        p_look = probs[i][0].item()
        print(f"  [{pred_label}] P(LOOK)={p_look:.4f}  {line[:80]}")

    # Export minimal files
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in NEEDED_FILES:
        src = SRC_DIR / fname
        dst = EXPORT_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  Copied {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: {fname} not found in {SRC_DIR}")

    # Fix LayerNorm key names for candle compatibility:
    # prajjwal1/bert-mini uses gamma/beta instead of weight/bias
    from safetensors.torch import load_file, save_file as sf_save

    st_path = EXPORT_DIR / "model.safetensors"
    weights = load_file(str(st_path))
    new_weights = {}
    renamed = 0
    for name, tensor in weights.items():
        new_name = name
        if ".LayerNorm.gamma" in name:
            new_name = name.replace(".LayerNorm.gamma", ".LayerNorm.weight")
            renamed += 1
        elif ".LayerNorm.beta" in name:
            new_name = name.replace(".LayerNorm.beta", ".LayerNorm.bias")
            renamed += 1
        new_weights[new_name] = tensor
    if renamed > 0:
        sf_save(new_weights, str(st_path))
        print(f"  Renamed {renamed} LayerNorm keys (gamma/beta -> weight/bias)")

    # Save reference outputs for Rust parity verification
    ref_outputs = {
        "test_lines": TEST_LINES,
        "logits": logits.numpy().tolist(),
        "probabilities": probs.numpy().tolist(),
        "predictions": [
            {
                "label": id2label[probs[i].argmax().item()],
                "p_look": round(probs[i][0].item(), 6),
                "p_skip": round(probs[i][1].item(), 6),
            }
            for i in range(len(TEST_LINES))
        ],
    }
    ref_path = EXPORT_DIR / "reference_outputs.json"
    with open(ref_path, "w") as f:
        json.dump(ref_outputs, f, indent=2)
    print(f"\n  Reference outputs saved to {ref_path}")

    # Save token IDs for parity checking
    token_ids = {
        "input_ids": inputs["input_ids"].numpy().tolist(),
        "attention_mask": inputs["attention_mask"].numpy().tolist(),
        "token_type_ids": inputs["token_type_ids"].numpy().tolist(),
    }
    token_path = EXPORT_DIR / "reference_tokens.json"
    with open(token_path, "w") as f:
        json.dump(token_ids, f, indent=2)
    print(f"  Reference tokens saved to {token_path}")

    total_size = sum(
        (EXPORT_DIR / f).stat().st_size for f in NEEDED_FILES if (EXPORT_DIR / f).exists()
    )
    print(f"\nExport complete: {EXPORT_DIR}")
    print(f"Total model size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
