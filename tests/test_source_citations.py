from redis import Redis

from common.get_source import get_triple_source
from pipeline_01_processing.pipeline import run_pipeline


def test_correct_node_source(db: Redis):
    """Should record source of node correctly."""

    source_text = (
        "High blood pressure is a common condition that affects the body's arteries. "
        "It's also called hypertension."
    )
    tripleset, source_id = run_pipeline(source_text)

    bp = tripleset.get_or_none(
        subject="high blood pressure", predicate="be", object="a common condition"
    )
    assert bp is not None

    # Check correct location coords
    assert bp.subject.loc == (source_id, 0, 2)
    assert bp.object.loc == (source_id, 4, 6)

    # Check text is saved to db
    res = db.get("source:" + source_id)
    assert res is not None
    assert str(res) == source_text

    # Check getting source from triple
    triple_source = get_triple_source(bp, sentences_before=0, sentences_after=0)
    assert (
        triple_source
        == "High blood pressure is a common condition that affects the body's arteries."
    )
