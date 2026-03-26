from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _source_keys(sources: Sequence[Dict], key: str = "id") -> List[str]:
    vals: List[str] = []
    for s in sources or []:
        v = s.get(key)
        if not v:
            # fallback to title if id missing
            v = s.get("title")
        vals.append(_normalize_str(str(v)))
    return vals


def precision_recall_at_k(
    gold: Sequence[str], retrieved: Sequence[str], k: int
) -> Tuple[float, float, int]:
    gset: Set[str] = { _normalize_str(x) for x in gold if _normalize_str(x) }
    rset: List[str] = [ _normalize_str(x) for x in retrieved[:k] if _normalize_str(x) ]
    if not rset:
        return (0.0, 0.0, 0)
    hit = len([x for x in rset if x in gset])
    prec = hit / len(rset) if rset else 0.0
    rec = hit / len(gset) if gset else 0.0
    return (prec, rec, hit)


def mrr_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    gset: Set[str] = { _normalize_str(x) for x in gold if _normalize_str(x) }
    for i, r in enumerate(retrieved[:k], start=1):
        if _normalize_str(r) in gset:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    # Binary relevance
    gset: Set[str] = { _normalize_str(x) for x in gold if _normalize_str(x) }
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        rel = 1.0 if _normalize_str(r) in gset else 0.0
        if rel:
            # log2 discount
            from math import log2

            dcg += rel / log2(i + 1)
    # Ideal DCG
    ideal_hits = min(len(gset), k)
    idcg = 0.0
    from math import log2

    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / log2(i + 1)
    return (dcg / idcg) if idcg else 0.0


def sources_nonempty_ratio(rows: Iterable[Dict]) -> float:
    rows = list(rows)
    total = len(rows)
    if total == 0:
        return 0.0
    nonempty = sum(1 for r in rows if r.get("retrieval_sources"))
    return nonempty / total

