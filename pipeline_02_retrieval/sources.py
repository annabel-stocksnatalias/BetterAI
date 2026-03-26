"""
sources.py
-----------
Module: RDF Result → Document Source Converter

Description:
    This module defines utilities to transform RDFLib SPARQL query results
    into a structured list of `DocSource` objects used in retrieval and
    hallucination-evaluation pipelines. It supports all major SPARQL query
    types (SELECT, CONSTRUCT, DESCRIBE) and is tailored for medical RDF
    graphs such as PubMed, Bio2RDF, UMLS, or Wikidata biomedical subsets.

Core Function:
    result_to_sources(res: rdflib.query.Result) -> list[DocSource]

Purpose:
    Converts RDF triples or tabular query outputs into a consistent internal
    representation (`DocSource`), simplifying evidence tracing and factuality
    evaluation in medical NLP systems.

----------------------------------------------------------------------
Example 1: SELECT Query
----------------------------------------------------------------------
SPARQL:
    SELECT ?uri ?label ?abstract
    WHERE {
        ?uri rdf:type :Drug .
        ?uri rdfs:label ?label .
        ?uri :hasDescription ?abstract .
    }
    LIMIT 2

Input (simplified):
    ?uri                                  ?label       ?abstract
    -----------------------------------   -----------  ------------------------------------------
    http://bio2rdf.org/drugbank:DB001     Aspirin      Used to treat pain, fever, and inflammation.
    http://bio2rdf.org/drugbank:DB002     Metformin    Controls blood sugar levels in diabetes.

Output:
    [
        DocSource(
            id="drugbank:DB001",
            title="Aspirin",
            content="Used to treat pain, fever, and inflammation.",
            source_type="SPARQL_SELECT"
        ),
        DocSource(
            id="drugbank:DB002",
            title="Metformin",
            content="Controls blood sugar levels in diabetes.",
            source_type="SPARQL_SELECT"
        )
    ]

----------------------------------------------------------------------
Example 2: CONSTRUCT Query
----------------------------------------------------------------------
SPARQL:
    CONSTRUCT { ?disease :treated_by ?drug . }
    WHERE { ?disease :treated_by ?drug . }
    LIMIT 2

Triples:
    (http://example.org/Disease#Diabetes, :treated_by, http://example.org/Drug#Insulin)
    (http://example.org/Disease#Influenza, :treated_by, http://example.org/Drug#Oseltamivir)

Output:
    [
        DocSource(
            id="Diabetes",
            title="treated_by",
            content="Insulin",
            source_type="RDF_TRIPLE"
        ),
        DocSource(
            id="Influenza",
            title="treated_by",
            content="Oseltamivir",
            source_type="RDF_TRIPLE"
        )
    ]

----------------------------------------------------------------------
Example 3: Empty Input
----------------------------------------------------------------------
Input:
    res = None

Output:
    []
----------------------------------------------------------------------
Notes:
    • No ML or LLM is required for this transformation — it is a
      deterministic mapping from structured RDF data to application-level
      document objects.
    • Designed for use in medical RAG and verification pipelines where RDF
      triples represent factual grounding.
"""


from typing import List

from rdflib.query import Result
from rdflib.term import BNode, Literal, URIRef

from .schemas.doc import DocSource


def result_to_sources(res: Result) -> List[DocSource]:
    """
    Convert an RDFLib SPARQL query `Result` into a structured list of `DocSource` items.
    Supports SELECT / CONSTRUCT / DESCRIBE result formats and safely extracts
    document identifiers, labels, abstracts, or URIs.

    Parameters
    ----------
    res : rdflib.query.Result
        The RDFLib query result to convert.

    Returns
    -------
    List[DocSource]
        A list of structured document sources ready for downstream use.
    """

    # --- 1 Validate input ---
    if res is None:
        return []

    docs: List[DocSource] = []

    # --- 2️ Handle CONSTRUCT/DESCRIBE queries ---
    if getattr(res, "type", None) in {"CONSTRUCT", "DESCRIBE"}:
        g = getattr(res, "graph", None)
        if g is None:
            return []
        for s, p, o in g:
            doc = DocSource(
                id=_format_term(s),
                title=_format_term(p),
                content=_format_term(o),
                source_type="RDF_TRIPLE",
                score=None,
            )
            docs.append(doc)
        return docs

    # --- 3 Handle SELECT queries ---
    try:
        variables = getattr(res, "vars", [])
        rows = list(res)

        if not rows:
            return []

        for row in rows:
            data = row.asdict() if hasattr(row, "asdict") else dict(zip(variables, row))

            # Common RDF columns
            uri = _format_term(data.get("uri") or data.get("id"))
            label = _format_term(data.get("label") or data.get("title"))
            abstract = _format_term(data.get("abstract") or data.get("description") or data.get("content"))

            # Create DocSource record
            doc = DocSource(
                id=uri or "unknown",
                title=label or "(untitled)",
                content=abstract or "",
                source_type="SPARQL_SELECT",
                score=None,
            )
            docs.append(doc)

    except Exception as e:
        print(f"[result_to_sources]  Error parsing RDF result: {e}")

    return docs


# -------------------------------------------------------------------
# Helper: Format RDF terms into readable strings
# -------------------------------------------------------------------

def _format_term(term) -> str:
    """Render an RDFLib term (URIRef, Literal, BNode) into clean text."""
    if term is None:
        return ""
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, URIRef):
        uri = str(term)
        # Use only the human-readable fragment of the URI
        return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    if isinstance(term, BNode):
        return f"_:{term}"
    return str(term)
