"""
Data Processing Pipline Entrypoint
"""

import uuid

from common.tokenize import tokenize_text
from database.rdf.tripleset import TripleSet
from database.redis.redis import get_redis_db

from .tokens_to_rdf import tokens_to_rdf


def run_pipeline(text: str) -> tuple[TripleSet, str]:
    """
    Main function for data processing pipeline.

    This pipeline takes in a long-form string of text and converts it to
    an RDF graph reprensentation.

    Parameters
    ----------
    text (str) : String of text used for processing.

    Returns
    -------
    0 (TripleSet) : Object in JSON-LD format representing the nodes created in the RDF database.
    1 (str) : Id of source text
    """

    # Step 1: Convert text to tokens using NER, POS, etc
    tokens = tokenize_text(text)

    source_id = uuid.uuid4().__str__()

    # Step 1.5: Save text to database
    with get_redis_db() as db:
        db.set(f"source:{source_id}", text)

    # Step 2: Convert tokens to RDF graph form
    graph = tokens_to_rdf(tokens, source_id=source_id)

    return (graph, source_id)
