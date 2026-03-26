"""
Train a simple ML baseline for PubMedQA yes/no classification.

Model:
  - Input: question + context text
  - Features: TF‑IDF (unigram + bigram)
  - Classifier: LogisticRegression (balanced)

Usage (from project root):
  uv run python -m evaluation.train_pubmedqa_baseline

This will write the trained model to:
  runs/models/pubmedqa_tfidf.joblib
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import numpy as np


DATA_PATH = Path("data/pubmedqa/pubmedqa.jsonl")
MODEL_PATH = Path("runs/models/pubmedqa_tfidf.joblib")


def load_pubmedqa_yesno(path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = (row.get("gold") or "").strip().lower()
            if gold not in {"yes", "no"}:
                continue
            q = row.get("question") or ""
            ctx = row.get("context") or ""
            text = f"{q} [SEP] {ctx}".strip()
            if not text:
                continue
            label = 1 if gold == "yes" else 0
            texts.append(text)
            labels.append(label)

    return texts, labels


def train_and_save() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"PubMedQA file not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    texts, labels = load_pubmedqa_yesno(DATA_PATH)
    print(f"Loaded {len(texts)} labeled samples.")

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline: Pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50000,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training TF‑IDF + LogisticRegression baseline...")
    pipeline.fit(X_train, y_train)

    # Validation metrics with default 0.5 threshold
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)
    acc_default = accuracy_score(y_val, y_pred_default)
    f1_default = f1_score(y_val, y_pred_default, average="macro")
    print(f"Validation (thr=0.50) Accuracy: {acc_default:.4f}")
    print(f"Validation (thr=0.50) Macro‑F1: {f1_default:.4f}")

    # Tune decision threshold for macro‑F1
    best_thr = 0.5
    best_f1 = f1_default
    # Use percentiles of the validation probabilities as candidate thresholds
    candidate_thrs = np.unique(np.percentile(y_proba, np.linspace(5, 95, 19)))
    for thr in candidate_thrs:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    print(f"Best validation threshold for Macro‑F1: {best_thr:.3f} (Macro‑F1={best_f1:.4f})")

    # Ensure output directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "threshold": best_thr}, MODEL_PATH)
    print(f"Saved model + threshold to: {MODEL_PATH}")


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
