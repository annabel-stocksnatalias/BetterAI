"""
Utilities to load PubMedQA-style JSONL.

Expected input fields per line (best-effort normalization):
  - id: str/int
  - question: str
  - answer or final_decision: 'yes'/'no' (case-insensitive)
  - context (optional): str
"""

from typing import Any, Dict, Iterable, List, Tuple

from .utils import read_jsonl, to_yes_no


def load_pubmedqa(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        q = row.get("question") or row.get("q") or row.get("Question")
        a = row.get("answer") or row.get("final_decision") or row.get("label") or row.get("gold")
        ctx = row.get("context") or row.get("abstract") or row.get("paragraph")
        rid = row.get("id") or row.get("qid") or row.get("pmid") or row.get("idx")
        if q is None:
            # skip incomplete lines (no question)
            continue
        data.append({
            "id": rid,
            "question": str(q),
            "gold": to_yes_no(str(a)) if a is not None else None,
            "context": str(ctx) if ctx is not None else "",
        })
    return data
