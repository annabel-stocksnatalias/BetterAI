"""
Schemas related to retrieval sources.

This schema is aligned with pipeline_02_retrieval/sources.py which
emits compact document-like records derived from RDF query results.
"""

import attr


@attr.s
class DocSource:
    """Represents a single retrieved source item.

    Notes
    -----
    - `id`, `title`, and `content` are human-readable fields derived from RDF terms.
    - `source_type` indicates the origin, e.g., "SPARQL_SELECT" or "RDF_TRIPLE".
    """

    id: str = attr.ib()
    title: str = attr.ib()
    content: str = attr.ib()
    source_type: str = attr.ib()
    # Optional confidence/relevance score (higher is better).
    score: float | None = attr.ib(default=None)
