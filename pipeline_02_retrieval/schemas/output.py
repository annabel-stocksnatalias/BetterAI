"""
Schemas related to the output of the pipeline.
"""

import attr

from .doc import DocSource


@attr.s
class Pipeline2Output:
    """Represents an object returned from Pipeline 2."""

    summary: str = attr.ib()
    """Plain-text summary provided to user."""

    sources: list[DocSource] = attr.ib()
    """Sources representing physical data used to generate summary."""

    grounded_answer: str | None = attr.ib(default=None)
    """Optional grounded answer with citations/refusal."""
