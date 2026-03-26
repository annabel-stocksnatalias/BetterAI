"""
Data Retrieval Pipline Entrypoint
"""

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Optional

from rdflib import Literal, URIRef
from rdflib.namespace import RDFS

from common.tokenize import tokenize_text
from database.rdf.rdf import RDFDatabase

# from pipeline_02_retrieval.patient_context import apply_patient_context
from pipeline_02_retrieval.generation import generate_grounded_answer
from pipeline_02_retrieval.schemas.doc import DocSource
from pipeline_02_retrieval.sources import result_to_sources
from pipeline_02_retrieval.summarize import result_to_summary

from .schemas.output import Pipeline2Output
from .tokens_to_query import tokens_to_query


def run_pipeline(db: RDFDatabase, text: str, patient_id: Optional[str] = None) -> Pipeline2Output:
    """
    Main function for data retrieval pipeline.

    Parameters
    ----------
    text (str) : String of text used for to create query.
    patient_id (str | None) : If provided, will apply query against patient context.

    Returns
    -------
    Output containing the answer summary and list of sources used to obtain summary.
    """

    # Step 1: Convert text to list of tokens using NER, POS, etc
    tokens = tokenize_text(text=text)

    # Step 2: Convert tokens to a structured query for database
    query = tokens_to_query(tokens=tokens)

    # Step 2.5: Optionally, apply patient context (disabled for now)
    # if patient_id:
    #     query = apply_patient_context(db, query=query, patient_id=patient_id)

    # Step 3: Execute query against the database
    try:
        res = db.graph.query(query)
    except Exception as e:
        # If the SPARQL query is malformed or execution fails, fall back to an empty result.
        # Log the offending query (truncated) to help with debugging.
        snippet = (query or "").replace("\n", " ")[:200]
        print(
            f"[Retrieval] SPARQL query failed; returning empty result. Error: {e}. Query snippet: {snippet!r}"
        )
        res = None

    # Step 4: Get text summary from query result
    summary = result_to_summary(res)

    # Step 5: Get document sources from query result
    sources = result_to_sources(res)
    sources = _score_sources(text, sources)
    sources = _dedupe_and_sort_sources(sources)

    # Step 5.5: If no sources found, try a lightweight lexical fallback
    if not sources:
        fallback_sources = _lexical_fallback_sources(db, text=text, limit=5)
        if fallback_sources:
            sources = _score_sources(text, fallback_sources)
            sources = _dedupe_and_sort_sources(sources)
            summary = f"Used lexical fallback; found {len(sources)} source(s)."
        else:
            summary = "Not enough evidence to answer the question."

    # Step 6: Produce a grounded answer with refusal if evidence is weak
    grounded_answer = generate_grounded_answer(question=text, sources=sources)

    # Step 7: Combine everything, return
    output = Pipeline2Output(summary=summary, sources=sources, grounded_answer=grounded_answer)

    return output


# ---------------------------------------------------------------------------
# Fallback retrieval: simple lexical scoring over labels and literals
# ---------------------------------------------------------------------------


def _lexical_fallback_sources(db: RDFDatabase, text: str, limit: int = 5) -> list[DocSource]:
    """
    When SPARQL yields nothing, fall back to a simple lexical search over the graph's
    labels and literal objects. Scores use a lightweight BM25-style weighting.
    """

    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", text or "") if len(t) > 2]
    if not tokens:
        return []

    # Build simple documents per subject: concatenate label + literals
    subj_docs: dict[URIRef, str] = defaultdict(str)
    subj_pred_obj: dict[URIRef, tuple[str, str]] = {}

    for subj, pred, obj in db.graph:
        if isinstance(obj, Literal):
            text_frag = str(obj)
            subj_docs[subj] += " " + text_frag
            subj_pred_obj[subj] = (str(pred), text_frag)
        if pred == RDFS.label and isinstance(obj, Literal):
            subj_docs[subj] += " " + str(obj)

    if not subj_docs:
        return []

    # Compute document lengths and DF for BM25-ish scoring
    doc_tokens: dict[URIRef, list[str]] = {}
    df: Counter[str] = Counter()
    for subj, doc_text in subj_docs.items():
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", doc_text) if len(t) > 2]
        doc_tokens[subj] = toks
        for t in set(toks):
            df[t] += 1

    N = len(doc_tokens)
    avgdl = sum(len(toks) for toks in doc_tokens.values()) / N if N else 0.0

    def bm25(term: str, freq: int, dl: int, k1: float = 1.5, b: float = 0.75) -> float:
        idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
        denom = freq + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
        return idf * ((freq * (k1 + 1)) / (denom or 1e-9))

    scores: dict[URIRef, float] = {}
    for subj, toks in doc_tokens.items():
        dl = len(toks) or 1
        tf = Counter(toks)
        score = 0.0
        for term in tokens:
            if term not in tf:
                continue
            score += bm25(term, tf[term], dl)
        if score > 0:
            scores[subj] = score

    if not scores:
        return []

    # Build DocSource objects sorted by score desc
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    docs: list[DocSource] = []
    for subj, score in ranked:
        pred_str, obj_text = subj_pred_obj.get(subj, ("", ""))
        docs.append(
            DocSource(
                id=str(subj).rsplit("/", 1)[-1],
                title=str(pred_str).rsplit("/", 1)[-1],
                content=obj_text,
                source_type="LEXICAL_FALLBACK",
                score=float(score),
            )
        )

    return docs


def _dedupe_and_sort_sources(sources: list[DocSource]) -> list[DocSource]:
    """Deduplicate sources by id/title/content/type, keeping the highest score and sorting by score."""
    best: dict[tuple, DocSource] = {}
    for s in sources or []:
        key = (s.id, s.title, s.content, s.source_type)
        existing = best.get(key)
        if existing is None:
            best[key] = s
        else:
            # Keep the one with higher score if available
            if (s.score or 0.0) > (existing.score or 0.0):
                best[key] = s

    deduped = list(best.values())
    deduped.sort(key=lambda s: (-(s.score or 0.0), s.source_type, s.id))
    return deduped


def _score_sources(query: str, sources: list[DocSource]) -> list[DocSource]:
    """Assign lexical scores to sources that lack a score, then optionally boost with embeddings."""
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9\\-]+", query or "") if len(t) > 2]
    tokset = set(tokens)

    for s in sources or []:
        if s.score is None:
            haystack = f"{s.title} {s.content}".lower()
            overlap = sum(1 for t in tokset if t in haystack)
            s.score = float(overlap)

    _maybe_embedding_score(query, sources)
    return sources


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Lazy-load an embedding model if available; return None otherwise."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _maybe_embedding_score(query: str, sources: list[DocSource]) -> None:
    """Boost scores using cosine similarity if sentence-transformers is available."""
    model = _get_embedding_model()
    if model is None or not sources:
        return

    try:
        import numpy as np
    except Exception:
        return

    query_text = (query or "").strip()
    if not query_text:
        return

    try:
        q_emb = model.encode([query_text], normalize_embeddings=True)[0]
    except Exception:
        return

    for s in sources:
        text = f"{s.title} {s.content}".strip()
        if not text:
            continue
        try:
            s_emb = model.encode([text], normalize_embeddings=True)[0]
            sim = float(np.dot(q_emb, s_emb))
            # If existing score is present, blend; otherwise set to similarity
            if s.score is None:
                s.score = sim
            else:
                s.score = float(s.score) + sim
        except Exception:
            continue
