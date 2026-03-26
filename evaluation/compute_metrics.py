"""
Aggregate metrics from prediction files.

Usage:
  python -m evaluation.compute_metrics --pred runs/pqa_eval.jsonl --gold-entities-key gold_mesh --gold-pmids-key gold_pmids --k 5
"""

import argparse
from typing import Any, Dict, Iterable, List

from .metrics import accuracy_and_macro_f1
from .retrieval_metrics import (
    _source_keys,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_at_k,
    sources_nonempty_ratio,
)
from .utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions JSONL from run_eval/judge")
    ap.add_argument("--gold-entities-key", default="gold_mesh", help="Key for gold entity IDs list")
    ap.add_argument("--gold-pmids-key", default="gold_pmids", help="Key for gold PMIDs list")
    ap.add_argument("--k", type=int, default=5, help="Cutoff k for retrieval metrics")
    args = ap.parse_args()

    rows = list(read_jsonl(args.pred))

    # QA metrics (only labeled)
    golds = [r.get("gold") for r in rows if r.get("gold") in {"yes", "no"}]
    preds = [r.get("pred") for r in rows if r.get("gold") in {"yes", "no"}]
    if golds and preds:
        acc, macro_f1 = accuracy_and_macro_f1(golds, preds)
        print(f"QA Samples: {len(golds)}  Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")
    else:
        print("QA: No labeled samples found; skipping Accuracy/F1.")

    # Retrieval metrics
    ratio = sources_nonempty_ratio(rows)
    print(f"Retrieval Coverage (non-empty sources): {ratio:.4f}")

    def get_gold_list(r: Dict[str, Any], key: str) -> List[str]:
        v = r.get(key)
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    # Entities-based metrics (if gold present)
    ent_prec_sum = ent_rec_sum = ent_mrr_sum = ent_ndcg_sum = 0.0
    ent_count = 0
    for r in rows:
        gold_entities = get_gold_list(r, args.gold_entities_key)
        if not gold_entities:
            continue
        retrieved_ids = _source_keys(r.get("retrieval_sources") or [], key="id")
        p, rec, _ = precision_recall_at_k(gold_entities, retrieved_ids, args.k)
        mrr = mrr_at_k(gold_entities, retrieved_ids, args.k)
        ndcg = ndcg_at_k(gold_entities, retrieved_ids, args.k)
        ent_prec_sum += p
        ent_rec_sum += rec
        ent_mrr_sum += mrr
        ent_ndcg_sum += ndcg
        ent_count += 1
    if ent_count:
        print(
            f"Entity Retrieval (k={args.k}) over {ent_count} samples: P@k={ent_prec_sum/ent_count:.4f} R@k={ent_rec_sum/ent_count:.4f} MRR={ent_mrr_sum/ent_count:.4f} NDCG={ent_ndcg_sum/ent_count:.4f}"
        )
    else:
        print("Entity Retrieval: No gold entity lists found; provide --gold-entities-key.")

    # PMID-based metrics (if gold present)
    pm_prec_sum = pm_rec_sum = pm_mrr_sum = pm_ndcg_sum = 0.0
    pm_count = 0
    for r in rows:
        gold_pmids = get_gold_list(r, args.gold_pmids_key)
        if not gold_pmids:
            continue
        retrieved_pmids = _source_keys(r.get("retrieval_sources") or [], key="id")
        p, rec, _ = precision_recall_at_k(gold_pmids, retrieved_pmids, args.k)
        mrr = mrr_at_k(gold_pmids, retrieved_pmids, args.k)
        ndcg = ndcg_at_k(gold_pmids, retrieved_pmids, args.k)
        pm_prec_sum += p
        pm_rec_sum += rec
        pm_mrr_sum += mrr
        pm_ndcg_sum += ndcg
        pm_count += 1
    if pm_count:
        print(
            f"PMID Retrieval (k={args.k}) over {pm_count} samples: P@k={pm_prec_sum/pm_count:.4f} R@k={pm_rec_sum/pm_count:.4f} MRR={pm_mrr_sum/pm_count:.4f} NDCG={pm_ndcg_sum/pm_count:.4f}"
        )
    else:
        print("PMID Retrieval: No gold pmid lists found; provide --gold-pmids-key.")

    # Faithfulness metrics if judgements exist
    total_claims = supported = contrad = nei = 0
    for r in rows:
        for c in r.get("claims", []) or []:
            total_claims += 1
            v = (c.get("verdict") or "").lower()
            if v == "supported":
                supported += 1
            elif v == "contradicted":
                contrad += 1
            else:
                nei += 1
    if total_claims:
        halluc = (contrad + nei) / total_claims
        fact_prec = supported / total_claims
        print(
            f"Faithfulness: Claims={total_claims} Supported={supported} Contradicted={contrad} NEI={nei} Hallucination={halluc:.4f} FactualPrecision={fact_prec:.4f}"
        )
    else:
        print("Faithfulness: No claim judgements found; run evaluation/judge_claims.py.")


if __name__ == "__main__":
    main()

