"""
CLI to run evaluation over a PubMedQA-style dataset.

Usage:
  python -m evaluation.run_eval --dataset path/to/dev.jsonl --output runs/pqa_eval.jsonl --model heuristic
  # or
  python evaluation/run_eval.py --dataset ...
"""

import argparse
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.baselines import always_yes, heuristic_yesno, ml_yesno, transformer_yesno
from evaluation.load_pubmedqa import load_pubmedqa
from evaluation.metrics import accuracy_and_macro_f1
from evaluation.utils import write_jsonl


def get_model(name: str):
    name = (name or "").lower()
    if name in {"always_yes", "yes"}:
        return always_yes
    if name in {"heuristic", "rule"}:
        return heuristic_yesno
    if name in {"ml", "tfidf", "ml_yesno"}:
        return ml_yesno
    if name in {"transformer", "hf"}:
        return transformer_yesno
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to PubMedQA-style JSONL")
    ap.add_argument("--output", required=True, help="Path to write predictions JSONL")
    ap.add_argument(
        "--model",
        default="heuristic",
        help="Model name: heuristic|always_yes|ml|transformer",
    )
    ap.add_argument("--with_retrieval", action="store_true", help="Run KG retrieval and save context")
    args = ap.parse_args()

    samples = load_pubmedqa(args.dataset)
    model = get_model(args.model)
    db = None

    outputs: List[Dict[str, Any]] = []
    for s in tqdm(samples, total=len(samples), desc="Evaluating"):
        row: Dict[str, Any] = {
            "id": s.get("id"),
            "question": s["question"],
            "gold": s["gold"],
        }
        if args.with_retrieval:
            # Lazy import to avoid dependency issues when not retrieving
            if db is None:
                from database.database import Database  # type: ignore
                db = Database()
            from evaluation.retrieval import retrieve_for_question  # type: ignore
            ret = retrieve_for_question(db, s["question"])
            row["retrieval_summary"] = ret.get("summary")
            row["retrieval_sources"] = ret.get("sources")
            # Enrich sample passed to the model with retrieval summary
            enriched_sample = dict(s)
            enriched_sample["retrieval_summary"] = ret.get("summary")
            pred = model(enriched_sample)
        else:
            pred = model(s)

        row["pred"] = pred
        row["correct"] = bool(s["gold"] == pred)
        outputs.append(row)

    # Metrics
    labeled_idx = [i for i, s in enumerate(samples) if s.get("gold") in {"yes", "no"}]
    if labeled_idx:
        acc, f1 = accuracy_and_macro_f1((samples[i]["gold"] for i in labeled_idx), (outputs[i]["pred"] for i in labeled_idx))
    else:
        acc, f1 = 0.0, 0.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, outputs)

    print(f"Samples: {len(samples)} (labeled: {len(labeled_idx)})")
    if labeled_idx:
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {f1:.4f}")
    else:
        print("No labels found; skipped accuracy/F1.")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
