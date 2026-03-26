from typing import Optional

from common.tokenize import tokenize_text
from database.rdf.triple import Triple
from database.redis.redis import get_redis_db


def get_source_text(
    source_id: str, start_index: int, end_index: int, sentences_before=1, sentences_after=1
):
    """Get source text based on source id and token locations."""

    source: Optional[str] = None

    with get_redis_db() as db:
        source = db.get("source:" + source_id)

    if source is None:
        return None

    doc = tokenize_text(source)

    start_sent = doc[start_index].sent
    end_sent = doc[end_index].sent

    start_token = doc[start_sent.start]
    end_token = doc[end_sent.end]

    # Get start location
    for _ in range(sentences_before):
        if start_token.i == 0:
            break

        prev_sentence = doc[start_token.i - 1].sent
        start_token = doc[prev_sentence.start]

    # Get end location
    for _ in range(sentences_after):
        if end_token.i == len(doc) - 1:
            break

        next_sentence = doc[end_token.i + 1].sent
        end_token = doc[next_sentence.end]

    return doc[start_token.i : end_token.i].text


def get_triple_source(triple: Triple, sentences_before=1, sentences_after=1):
    """Get source text surrounding triple location."""

    if not triple.predicate.loc:
        return None

    # Location of triple is stored on the predicate
    source_id, start, end = triple.predicate.loc

    return get_source_text(
        source_id, start, end, sentences_before=sentences_before, sentences_after=sentences_after
    )
