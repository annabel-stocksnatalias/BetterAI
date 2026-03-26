import os
import re
from typing import Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDFS

from .triple import Triple
from .tripleset import TripleSet


class RDFDatabase:
    """Utility class to manage RDF graph."""

    instance: Optional["RDFDatabase"] = None
    graph: Graph

    def __new__(cls) -> "RDFDatabase":
        if not cls.instance:
            cls.instance = super().__new__(cls)

            graph = Graph()
            # Attempt to load an existing graph if available
            if os.path.exists("./graph.json"):
                try:
                    graph.parse("./graph.json", format="json-ld")
                except Exception:
                    # Ignore load failures; start with an empty graph
                    pass

            # Best-effort initial serialize; ignore failures (e.g., read-only FS)
            try:
                graph.serialize(destination="./graph.json", format="json-ld")
            except Exception:
                pass

            cls.instance.graph = graph

        return cls.instance

    def apply_tripleset(self, payload: TripleSet):
        """Apply a simple list of triple-like dicts to the RDF graph.

        Expected input: list of dicts with keys {"s", "p", "o"}.
        This is a pragmatic bridge from Pipeline 1 output to RDFLib.
        """

        if not payload:
            return

        for item in payload:
            if not isinstance(item, Triple):
                continue
            s_raw = item.subject
            p_raw = item.predicate
            o_raw = item.object

            if not (s_raw and p_raw and o_raw):
                continue

            s = s_raw.to_rdf()
            p = p_raw.to_rdf()
            o = o_raw.to_rdf()

            self.graph.add((s, p, o))
            # Expose subject as an rdfs:label for label-based binding in retrieval
            try:
                self.graph.add((s, RDFS.label, Literal(str(s_raw))))
                self.graph.add((o, RDFS.label, Literal(str(o_raw))))
            except Exception:
                pass

        # Best-effort persist; ignore failures in restricted environments
        try:
            self.graph.serialize(destination="./graph.json", format="json-ld")
        except Exception:
            pass

    def apply_json(self, payload: list[dict]):
        """Apply a simple list of triple-like dicts (JSON-style) to the RDF graph.

        Expected input: list of dicts with keys {"s", "p", "o"}.
        This is a pragmatic bridge from ingestion pipelines to RDFLib.
        """

        if not payload:
            return

        NS = Namespace("http://example.org/node/")
        REL = Namespace("http://example.org/rel/")

        def slug(text: str) -> str:
            text = str(text or "").strip().lower()
            text = re.sub(r"\s+", "_", text)
            text = re.sub(r"[^a-z0-9_\-]", "", text)
            return text or "unnamed"

        for item in payload:
            if not isinstance(item, dict):
                continue
            s_raw = item.get("s")
            p_raw = item.get("p")
            o_raw = item.get("o")
            label_raw = item.get("label") or s_raw

            if not (s_raw and p_raw and o_raw):
                continue

            s = URIRef(NS + slug(s_raw))
            p = URIRef(REL + slug(p_raw))
            # Store object as literal for simplicity
            o = Literal(str(o_raw))

            self.graph.add((s, p, o))
            # Also add a best-effort rdfs:label for the subject to enable label binding
            try:
                if label_raw:
                    self.graph.add((s, RDFS.label, Literal(str(label_raw))))
            except Exception:
                pass

        # Best-effort persist; ignore failures in restricted environments
        try:
            self.graph.serialize(destination="./graph.json", format="json-ld")
        except Exception:
            pass
