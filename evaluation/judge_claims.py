"""
Heuristic claim extraction and judging against retrieved sources.

This is a lightweight placeholder to enable faithfulness metrics without LLMs.
It extracts simple sentence-like claims from the model's answer text and marks
them Supported if a majority of content words appear in any source title/content.
Otherwise, Not-Enough-Info (NEI). No Contradicted logic here.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List

from .utils import read_jsonl, write_jsonl


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def split_claims(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Split on sentence boundaries crudely
    parts = re.split(r"[.!?]+\s+", text)
    claims = [p.strip() for p in parts if p.strip()]
    return claims


def content_words(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(s)]


NEG_CUES = {"no", "not", "never", "without", "lack", "lacks", "absent", "absence", "neither"}


def _has_negation(text: str) -> bool:
    return any(f" {cue} " in f" {text.lower()} " for cue in NEG_CUES)


def judge_claim_against_sources(claim: str, sources: List[Dict[str, Any]], min_overlap: int = 3) -> str:
    cwords = set(content_words(claim))
    if not cwords:
        return "NEI"

    claim_neg = _has_negation(claim)

    verdict = "NEI"
    for src in sources or []:
        text = f"{src.get('title','')} {src.get('content','')}"
        swords = set(content_words(text))
        overlap = len(cwords & swords)
        if overlap < min_overlap:
            continue

        src_neg = _has_negation(text)
        if claim_neg != src_neg:
            return "Contradicted"
        verdict = "Supported"

    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Predictions JSONL from run_eval")
    ap.add_argument("--output", required=True, help="Output JSONL with claim judgments")
    ap.add_argument("--answer-key", default="pred", help="Field name for model answer text")
    args = ap.parse_args()

    rows = list(read_jsonl(args.input))
    out_rows: List[Dict[str, Any]] = []

    for r in rows:
        answer = str(r.get(args.answer_key) or "").strip()
        sources = r.get("retrieval_sources") or []
        claims = split_claims(answer)
        judgments = []
        for c in claims:
            verdict = judge_claim_against_sources(c, sources)
            judgments.append({"claim": c, "verdict": verdict})
        r_out = dict(r)
        r_out["claims"] = judgments
        out_rows.append(r_out)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, out_rows)

    # Aggregate metrics
    total_claims = sum(len(r.get("claims", [])) for r in out_rows)
    supported = sum(1 for r in out_rows for c in r.get("claims", []) if c.get("verdict") == "Supported")
    nei = sum(1 for r in out_rows for c in r.get("claims", []) if c.get("verdict") == "NEI")
    hallucination_rate = (nei / total_claims) if total_claims else 0.0
    factual_precision = (supported / total_claims) if total_claims else 0.0
    print(f"Claims: {total_claims}")
    print(f"Supported: {supported}")
    print(f"NEI: {nei}")
    print(f"Hallucination Rate: {hallucination_rate:.4f}")
    print(f"Factual Precision: {factual_precision:.4f}")


if __name__ == "__main__":
    main()
