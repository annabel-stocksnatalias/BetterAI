"""
RDF Medical Query Summarizer
----------------------------
Transforms an RDFLib SPARQL query `Result` object into an easy-to-read summary,
optimized for medical knowledge graphs (e.g., disease–drug–gene–symptom triples).

# Example Output
Found 3 record(s).
Variables: disease, treatment

Showing first 3 record(s):
  1. disease: Diabetes, treatment: Insulin
  2. disease: Hypertension, treatment: Atenolol
  3. disease: Influenza, treatment: Oseltamivir

Extracted Medical Entities:
  - Disease: Diabetes, Hypertension, Influenza
  - Drug: Atenolol, Insulin, Oseltamivir


"""

from typing import Any, Dict, List, Optional

from rdflib.query import Result
from rdflib.term import BNode, Literal, URIRef


def summarize_rdf_result(result: Optional[Result]) -> str:
    """
    Generate a text summary of an RDFLib SPARQL Result.
    Handles SELECT, ASK, CONSTRUCT, and DESCRIBE query outputs.
    Tailored for summarizing medical RDF triples.
    """

    # --- Step 1: Handle missing or invalid result ---
    if result is None:
        return "No RDF result object provided."

    # --- Step 2: Handle boolean (ASK) queries ---
    if getattr(result, "type", None) == "ASK":
        verdict = "True" if bool(result) else "False"
        return f"ASK query result: {verdict}"

    # --- Step 3: Handle graph-like results (CONSTRUCT / DESCRIBE) ---
    if getattr(result, "type", None) in {"CONSTRUCT", "DESCRIBE"}:
        g = getattr(result, "graph", None)
        triples = list(g) if g is not None else []
        if not triples:
            return "No RDF triples returned."

        output = [f"Retrieved {len(triples)} RDF triple(s). Showing up to 5:"]
        for i, (subj, pred, obj) in enumerate(triples[:5], 1):
            output.append(f"  {i}. {format_term(subj)} — {format_term(pred)} → {format_term(obj)}")

        if len(triples) > 5:
            output.append(f"  ... and {len(triples) - 5} additional triple(s).")

        return "\n".join(output)

    # --- Step 4: Handle tabular SELECT query results ---
    try:
        variables = getattr(result, "vars", [])
        records = list(result)

        if not records:
            return "The SELECT query returned no results."

        summary_lines = [f"🩺 Found {len(records)} record(s)."]
        if variables:
            summary_lines.append(f"Variables: {', '.join(map(str, variables))}")

        limit = min(5, len(records))
        summary_lines.append(f"\nShowing first {limit} record(s):")

        # Format each record (row) cleanly
        for idx, record in enumerate(records[:limit], start=1):
            data = record.asdict() if hasattr(record, "asdict") else dict(zip(variables, record))
            formatted_pairs = [f"{var}: {format_term(data.get(var))}" for var in variables]
            summary_lines.append(f"  {idx}. {', '.join(formatted_pairs)}")

        if len(records) > limit:
            summary_lines.append(f"  ... and {len(records) - limit} more result(s).")

        # --- Step 5: Optional medical entity extraction ---
        categories = detect_medical_entities(records, variables)
        if categories:
            summary_lines.append("\nExtracted Medical Entities:")
            for label, terms in categories.items():
                summary_lines.append(f"  - {label}: {', '.join(sorted(terms))}")

        return "\n".join(summary_lines)

    except Exception as error:
        return f"Failed to summarize RDF result: {error}"


# Backwards-compatible shim for existing imports
def result_to_summary(result: Optional[Result]) -> str:
    """Compatibility wrapper that delegates to summarize_rdf_result."""
    return summarize_rdf_result(result)


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def format_term(term: Any) -> str:
    """
    Convert RDFLib terms (URIRef, Literal, BNode) into compact, readable text.
    """
    if term is None:
        return "None"
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, URIRef):
        uri_str = str(term)
        # Keep only the meaningful part (last fragment of URI)
        return uri_str.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    if isinstance(term, BNode):
        return f"_:{term}"
    return str(term)


def detect_medical_entities(records: List[Any], variables: List[str]) -> Dict[str, List[str]]:
    """
    Identify and group medically relevant entities from RDF query results
    (e.g., diseases, drugs, genes, symptoms).
    """
    keyword_map = {
        "Disease": ["disease", "syndrome", "disorder", "infection", "cancer", "flu"],
        "Drug": ["drug", "compound", "insulin", "aspirin", "therapy", "treatment"],
        "Gene": ["gene", "protein", "enzyme", "mutation"],
        "Symptom": ["symptom", "pain", "fever", "nausea"],
    }

    detected: Dict[str, set] = {k: set() for k in keyword_map}

    for rec in records:
        data = rec.asdict() if hasattr(rec, "asdict") else dict(zip(variables, rec))
        for val in data.values():
            if not val:
                continue
            text = str(val).lower()
            for category, cues in keyword_map.items():
                if any(cue in text for cue in cues):
                    detected[category].add(format_term(val))

    # Filter out empty groups
    return {cat: sorted(list(vals)) for cat, vals in detected.items() if vals}
