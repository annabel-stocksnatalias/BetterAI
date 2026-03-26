import uuid

from common.tokenize import tokenize_text
from pipeline_01_processing.tokens_to_rdf import tokens_to_rdf


def test_alias_extraction():
    """Should extract aliases from text."""

    source_text = "Hypertension (HTN) is a medical condition."
    source_id = uuid.uuid4()

    doc = tokenize_text(source_text)
    tripleset = tokens_to_rdf(doc, source_id)

    # Register alias from parenthesis
    res_1 = tripleset.get_or_none(subject="Hypertension", predicate="alias", object="HTN")
    assert res_1 is not None

    # Reverse association for alias
    res_2 = tripleset.get_or_none(subject="HTN", predicate="alias for", object="Hypertension")
    assert res_2 is not None

    # Direct access via root
    res_3 = tripleset.get_or_none(
        subject="Hypertension", predicate="be", object="a medical condition"
    )
    assert res_3 is not None

    # Allow access to connected nodes to root
    res_4 = tripleset.get_or_none(
        subject="HTN", predicate="be", object="a medical condition", get_root=True
    )
    assert res_4 is not None


def test_alias_assign_root_after_context():
    """Should assign objects to the root when alias defined after root."""

    source_text = "Hypertension (HTN) is a medical condition. HTN affects the body's arteries."
    source_id = uuid.uuid4()

    doc = tokenize_text(source_text)
    tripleset = tokens_to_rdf(doc, source_id)

    # Normal assignment
    res_1 = tripleset.get_or_none(
        subject="Hypertension", predicate="be", object="a medical condition"
    )
    assert res_1 is not None

    # Assignment via alias
    res_2 = tripleset.get_or_none(
        subject="Hypertension", predicate="affect", object="the body's artery"
    )
    assert res_2 is not None


# TODO: Account for this edge case
# def test_alias_assign_root_before_context():
#     """Should reassign objects to the root even if root defined after alias."""

#     source_text = (
#         "HTN is a medical condition. HTN, short for Hypertension, affects the body's arteries."
#     )

#     doc = tokenize_text(source_text)
#     tripleset = tokens_to_rdf(doc)

#     # Re-associate with root
#     res_1 = tripleset.get_or_none(
#         subject="Hypertension", predicate="be", object="a medical condition"
#     )
#     assert res_1 is not None

#     # Un-associate from original node
#     res_2 = tripleset.get_or_none(subject="HTN", predicate="be", object="a medical condition")
#     assert res_2 is None

#     # Allow access to node via root
#     res_3 = tripleset.get_or_none(
#         subject="HTN", predicate="be", object="a medical condition", get_root=True
#     )
#     assert res_3 is not None

#     # Indirect assignment to root
#     res_4 = tripleset.get_or_none(
#         subject="Hypertension", predicate="affect", object="the body's artery"
#     )
#     assert res_4 is not None
