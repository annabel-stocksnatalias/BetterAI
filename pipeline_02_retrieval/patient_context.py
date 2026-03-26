from rdflib.plugins.sparql.sparql import Query

from database.database import Database


def apply_patient_context(db: Database, query: str | Query, patient_id: str) -> str | Query:
    """Apply patient context to query.

    Minimal pass-through for now; hook for future FILTER/GRAPH clauses.
    """

    return query
