from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDFS

from pipeline_02_retrieval.pipeline import run_pipeline


class DummyDb:
    """Minimal DB stub exposing an RDFLib graph."""

    def __init__(self):
        self.graph = Graph()


def seed_graph_with_hypertension(db: DummyDb):
    ns = Namespace("http://example.org/node/")
    rel = Namespace("http://example.org/rel/")
    subj = URIRef(ns + "hypertension")

    # Label for subject binding
    db.graph.add((subj, RDFS.label, Literal("hypertension")))
    # Simple predicates expected by ingestion/prefix scheme
    db.graph.add((subj, rel["has_title"], Literal("Hypertension Overview")))
    db.graph.add((subj, rel["has_abstract"], Literal("High blood pressure condition.")))


def test_retrieval_pipeline_returns_sources_from_labeled_subject():
    db = DummyDb()
    seed_graph_with_hypertension(db)

    out = run_pipeline(db=db, text="What is hypertension?")

    assert out.sources, "Expected at least one source from retrieval"
    assert "found" in out.summary.lower(), f"Unexpected summary: {out.summary}"
    assert all(s.score is not None for s in out.sources), "Expected sources to be scored"


def test_retrieval_pipeline_uses_fallback_when_sparql_empty():
    db = DummyDb()
    seed_graph_with_hypertension(db)

    # Query text unlikely to match SPARQL intent, forcing lexical fallback
    out = run_pipeline(db=db, text="Explain high blood pressure condition")

    assert out.sources, "Expected fallback sources when SPARQL is empty"
    assert any(s.source_type == "LEXICAL_FALLBACK" for s in out.sources)
    assert out.grounded_answer, "Grounded answer should be populated"
    # Sources should be sorted by score (if available)
    scores = [s.score for s in out.sources if s.score is not None]
    assert scores == sorted(scores, reverse=True), "Sources are not sorted by score"


def test_retrieval_pipeline_refuses_when_no_evidence():
    db = DummyDb()

    out = run_pipeline(db=db, text="What is hypertension?")

    assert not out.sources, "No sources should be found on empty graph"
    assert "not enough evidence" in out.summary.lower()
    assert out.grounded_answer
    assert "not enough evidence" in out.grounded_answer.lower()
