"""
Fine-tune a small transformer (e.g., BioBERT/PubMedBERT) on PubMedQA.

Training signal:
  - Labels: yes/no from gold field
  - Inputs (default): question + context
  - Optional: add retrieval_summary from a prior KG run:
      runs/pubmedqa_eval_ml_full_with_ret.jsonl

Usage (from project root):
  uv run python -m evaluation.train_pubmedqa_transformer

This will write the fine-tuned model to:
  runs/models/pubmedqa_transformer
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_DATA_PATH = Path("data/pubmedqa/pubmedqa.jsonl")
ARTIFICIAL_PATH = Path("data/pubmedqa/pubmedqa_artificial.jsonl")
EVAL_WITH_RET_PATH = Path("runs/pubmedqa_eval_ml_full_with_ret.jsonl")
OUTPUT_DIR = Path("runs/models/pubmedqa_transformer")


def load_base_pubmedqa(path: Path) -> Dict[str, Dict]:
    by_id: Dict[str, Dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = str(row.get("id") or "").strip()
            if not rid:
                continue
            by_id[rid] = row
    return by_id


def build_examples_with_retrieval(
    base_by_id: Dict[str, Dict],
    eval_with_ret_path: Path,
) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    with eval_with_ret_path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = str(row.get("id") or "").strip()
            gold = (row.get("gold") or "").strip().lower()
            if gold not in {"yes", "no"} or not rid:
                continue

            base = base_by_id.get(rid)
            if not base:
                continue

            question = base.get("question") or ""
            context = base.get("context") or ""
            retrieval_summary = row.get("retrieval_summary") or ""

            parts = [
                f"Question: {question.strip()}",
                f"Context: {context.strip()}",
            ]
            if retrieval_summary:
                parts.append(f"Retrieval: {retrieval_summary.strip()}")

            text = " [SEP] ".join(parts)
            label = 1 if gold == "yes" else 0

            texts.append(text)
            labels.append(label)

    return texts, labels


def load_artificial(path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    if not path.exists():
        return texts, labels

    with path.open() as f:
        for line in f:
            row = json.loads(line)
            gold = (row.get("gold") or "").strip().lower()
            if gold not in {"yes", "no"}:
                continue
            question = row.get("question") or ""
            context = row.get("context") or ""
            text = f"Question: {question.strip()} [SEP] Context: {context.strip()}"
            label = 1 if gold == "yes" else 0
            texts.append(text)
            labels.append(label)

    return texts, labels


@dataclass
class TextLabelDataset(Dataset):
    texts: List[str]
    labels: List[int]
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        enc["labels"] = label
        return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="HF model name to fine-tune",
    )
    ap.add_argument(
        "--use_artificial",
        action="store_true",
        help="Include pubmedqa_artificial.jsonl as additional training data",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train/eval batch size",
    )
    args = ap.parse_args()

    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(f"Base PubMedQA file not found at {BASE_DATA_PATH}")
    if not EVAL_WITH_RET_PATH.exists():
        raise FileNotFoundError(
            f"Eval-with-retrieval file not found at {EVAL_WITH_RET_PATH}. "
            "Run `uv run evaluation/run_eval.py --dataset data/pubmedqa/pubmedqa.jsonl "
            "--output runs/pubmedqa_eval_ml_full_with_ret.jsonl --model ml --with_retrieval` first."
        )

    print(f"Loading base PubMedQA from {BASE_DATA_PATH}...")
    base_by_id = load_base_pubmedqa(BASE_DATA_PATH)

    print(f"Building examples with retrieval from {EVAL_WITH_RET_PATH}...")
    texts_main, labels_main = build_examples_with_retrieval(base_by_id, EVAL_WITH_RET_PATH)
    print(f"Main set with retrieval: {len(texts_main)} examples.")

    texts: List[str] = list(texts_main)
    labels: List[int] = list(labels_main)

    if args.use_artificial and ARTIFICIAL_PATH.exists():
        print(f"Loading artificial PubMedQA from {ARTIFICIAL_PATH}...")
        texts_art, labels_art = load_artificial(ARTIFICIAL_PATH)
        print(f"Artificial set: {len(texts_art)} examples.")
        texts.extend(texts_art)
        labels.extend(labels_art)

    print(f"Total training examples: {len(texts)}")

    # Simple train/val split
    split_idx = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    train_ds = TextLabelDataset(train_texts, train_labels, tokenizer)
    val_ds = TextLabelDataset(val_texts, val_labels, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels_eval = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels_eval, preds)
        f1 = f1_score(labels_eval, preds, average="macro")
        return {"accuracy": acc, "macro_f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting transformer fine-tuning...")
    trainer.train()

    print("Saving final model and tokenizer...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

