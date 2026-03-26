"""
Full integration tests for tokenization.
"""

import uuid

from common.tokenize import tokenize_text
from pipeline_01_processing.tokens_to_rdf import tokens_to_rdf


def test_generate_noun_attributes():
    """Should generate noun chunks."""

    source_text = (
        "High blood pressure is a common condition that affects the body's arteries. "
        "It's also called hypertension."
    )
    source_id = uuid.uuid4()

    doc = tokenize_text(source_text)
    tripleset = tokens_to_rdf(doc, source_id=source_id)

    bp = tripleset.filter(subject="high blood pressure")
    assert bp.count() >= 3, bp

    t_condition = bp.get_or_none(predicate="be", object="a common condition")
    assert t_condition is not None

    t_hypertension = bp.get_or_none(predicate="call", object="hypertension")
    assert t_hypertension is not None

    t_affects = bp.get_or_none(predicate="affect", object="the body's artery")
    assert t_affects is not None


# # TODO: Implement metrics resolution
# def test_metrics_resolution():
#     """Should resolve assignment statements that involve numerical values."""

#     # Ref: https://www.ncbi.nlm.nih.gov/books/NBK539859/
#     source_text = (
#         "The current definition of hypertension (HTN) is systolic blood pressure (SBP) values of 130 mm Hg "
#         "or more and/or diastolic blood pressure (DBP) of more than 80 mm Hg. Hypertension ranks among the most "
#         "common chronic medical condition characterized by a persistent elevation in arterial pressure."
#     )
#     source_id = uuid.uuid4()

#     doc = tokenize_text(source_text)
#     tripleset = tokens_to_rdf(doc, source_id)

#     # Check aliases
#     hypertension = tripleset.get_or_none(subject="hypertension", predicate="alias", object="HTN")
#     assert hypertension is not None

#     systolic = tripleset.get_or_none(
#         subject="systolic blood pressure", predicate="alias", object="SBP"
#     )
#     assert systolic is not None

#     diastolic = tripleset.get_or_none(
#         subject="diastolic blood pressure", predicate="alias", object="DBP"
#     )
#     assert diastolic is not None

#     # Check units
#     systolic_unit = tripleset.get_or_none(
#         subject="systolic blood pressure", predicate="unit", object="mm Hg"
#     )
#     assert systolic_unit is not None

#     diastolic_unit = tripleset.get_or_none(
#         subject="diastolic blood pressure", predicate="unit", object="mm Hg"
#     )
#     assert diastolic_unit is not None

#     # Check metrics
#     hypertension_set = tripleset.filter(subject="hypertension")

#     t_sys = hypertension_set.get_or_none(predicate="systolic blood pressure", object=130)
#     assert t_sys is not None

#     t_dia = hypertension_set.get_or_none(predicate="diastolic blood pressure", object=80)
#     assert t_dia is not None
