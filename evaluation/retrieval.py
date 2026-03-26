from typing import Any, Dict


def retrieve_for_question(db, question: str) -> Dict[str, Any]:
    """Run retrieval pipeline and return a compact dict for eval storage.

    Lazy-imports heavy dependencies to keep baseline runs lightweight.
    """

    from pipeline_02_retrieval.pipeline import run_pipeline as run_retrieval_pipeline  # local import

    out = run_retrieval_pipeline(db=db, text=question)
    # Convert sources to plain dicts for JSON output
    sources = [
        {
            "id": s.id,
            "title": s.title,
            "content": s.content,
            "source_type": s.source_type,
            "score": s.score,
        }
        for s in (out.sources or [])
    ]
    return {
        "summary": out.summary,
        "grounded_answer": getattr(out, "grounded_answer", None),
        "sources": sources,
    }
