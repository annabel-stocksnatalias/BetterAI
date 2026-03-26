"""
Grounded answer generator with refusal logic.

This module provides a minimal, model-agnostic way to turn retrieval sources
into a cited answer. It is intentionally simple to avoid adding heavy
dependencies; swap the body of `generate_grounded_answer` with an LLM call if
available.
"""

from typing import Iterable

from .schemas.doc import DocSource


def generate_grounded_answer(question: str, sources: Iterable[DocSource], score_threshold: float = 0.0) -> str:
    """
    Produce a grounded answer string with inline citations from a list of sources.
    If there are no sources or all scores are below threshold, return a refusal.
    """
    src_list = list(sources or [])
    if not src_list:
        return "Not enough evidence to answer the question based on available sources."

    # Filter low-score sources if scores are provided
    filtered = []
    for s in src_list:
        if s.score is None or s.score >= score_threshold:
            filtered.append(s)
    if not filtered:
        return "Not enough high-confidence evidence to answer the question."

    lines = []
    for idx, s in enumerate(filtered, start=1):
        citation = f"[{idx}] {s.title or s.id}"
        content = s.content or ""
        lines.append(f"{citation}: {content}")

    # Simple stitched answer; replace with LLM call if desired.
    answer = f"Based on the retrieved evidence, here is a summary for: {question.strip()}\n" + "\n".join(lines)
    return answer
