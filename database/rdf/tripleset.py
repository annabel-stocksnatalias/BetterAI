from collections.abc import Callable
from typing import Optional, TypeVar

import attrs
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from .triple import Node, Pred, Slot, SlotLike, SlotLoc, Triple

T = TypeVar("T")


@attrs.define
class TripleSet:
    triples: list[Triple]
    nodes: dict[str, Node] = {}

    def __str__(self):
        return f"[{', '.join([triple.__str__() for triple in self.triples])}]"

    def __iter__(self):
        return iter(self.triples)

    # ------------------------------------
    # Private Utility Methods
    # ------------------------------------

    @staticmethod
    def _slot_query(
        callable: Callable[["TripleSet", Optional[Slot], Optional[Slot], Optional[Slot], bool], T],
    ):
        """Allow input of raw values into function, and convert them to slots."""

        def wrapper(
            self,
            subject: Optional[SlotLike] = None,
            predicate: Optional[SlotLike] = None,
            object: Optional[SlotLike] = None,
            get_root: bool = False,
        ) -> T:

            if not isinstance(subject, Slot) and subject is not None:
                subject = Slot(subject)

            if not isinstance(predicate, Slot) and predicate is not None:
                predicate = Slot(predicate)

            if not isinstance(object, Slot) and object is not None:
                object = Slot(object)

            # subject = Slot(subject) if subject is not None else None
            # predicate = Slot(predicate) if predicate is not None else None
            # object = Slot(object) if object is not None else None

            return callable(self, subject, predicate, object, get_root=get_root)

        return wrapper

    def _get_root_subject(self, subject: Slot) -> Slot:
        """Get the root triple for a subject."""

        alias = self.get_or_none(predicate="alias", object=subject)

        if not alias:
            return subject

        while True:
            new_alias = self.get_or_none(predicate="alias", object=alias)
            if not new_alias:
                break
            else:
                alias = new_alias

        return alias.subject

    # ------------------------------------
    # Query Methods
    # ------------------------------------

    @_slot_query
    def get_or_none(
        self,
        subject: Optional[Slot] = None,
        predicate: Optional[Slot] = None,
        object: Optional[Slot] = None,
        get_root=False,
    ) -> Triple | None:
        """
        Find first matching triple or none.

        If root=True, will swap out subject for the top most parent node
        if applicable. Ex: "HTN" would be replaced with "hypertension".
        """

        assert (
            subject is not None or predicate is not None or object is not None
        ), "Must provide a subject, predicate, and/or object"

        if get_root is True and subject is not None:
            subject = self._get_root_subject(subject)

        return next(
            (triple for triple in self.triples if triple == (subject, predicate, object)), None
        )

    @_slot_query
    def filter(
        self,
        subject: Optional[Slot] = None,
        predicate: Optional[Slot] = None,
        object: Optional[Slot] = None,
        get_root=False,
    ) -> "TripleSet":
        """Return new TripleSet with triples that match query."""

        if get_root is True:
            subject = self._get_root_subject(subject)

        filtered_triples = [
            triple for triple in self.triples if triple == (subject, predicate, object)
        ]

        return TripleSet(filtered_triples)

    def count(self):
        """Get number of triples in tripleset."""

        return len(self.triples)

    # ------------------------------------
    # Management Methods
    # ------------------------------------

    def create_predicate(self, text: Token | Span | str):
        """Create predicate slot for triple."""

        if isinstance(text, Token) or isinstance(text, Span):
            text = text.lemma_
        else:
            text = str(text)

        return Pred(text)

    def get_or_create_node(self, text: Token | Span | str, source_id: Optional[str] = None):
        """Get existing node, or create new node given text."""

        start: Optional[int] = None
        end: Optional[int] = None
        loc: Optional[SlotLoc] = None

        # Replace with noun chunk if needed
        if isinstance(text, Token) and text._.noun_chunk is not None:
            text = text._.noun_chunk

        # Get final text, start, and end
        if isinstance(text, Token):
            start = text.idx
            end = text.idx
            text = text.lemma_
        elif isinstance(text, Span):
            start = text.start
            end = text.end - 1  # Spacy gives index of token after span
            text = text.lemma_
        else:
            text = str(text)

        if source_id is not None and start is not None:
            loc = (source_id, start, end)

        if text.lower() not in self.nodes.keys():
            self.nodes[text] = Node(text, loc=loc)

        return self.nodes[text]

    def create_triple(
        self,
        subject: Node,
        predicate: Pred,
        object: Node,
        get_root=True,
        loc: Optional[SlotLoc] = None,
    ):
        """Creates a new rdf triple."""

        if get_root:
            subject = self._get_root_subject(subject)

        if loc:
            predicate.loc = loc

        triple = Triple(subject, predicate, object)
        self.triples.append(triple)

        return triple

    # def get_or_create(
    #     self,
    #     subject: Token | SlotPrimitive,
    #     predicate: Token | SlotPrimitive,
    #     object: Token | SlotPrimitive,
    #     get_root=True,
    #     loc: Optional[SlotLoc] = None,
    # ):
    #     """Get one or create a new triple."""

    #     if isinstance(subject, Token) or isinstance(subject, Span):
    #         subject = subject.lemma_

    #     if isinstance(predicate, Token or isinstance(predicate, Span)):
    #         predicate = predicate.lemma_

    #     if isinstance(object, Token) or isinstance(object, Span):
    #         object = object.lemma_

    #     if get_root:
    #         subject = self._get_root_subject(Slot(subject))

    #     triple = self.get_or_none(subject, predicate, object)

    #     if not triple:
    #         triple = Triple(subject, predicate, object)
    #         self.triples.append(triple)

    #     return triple

    # def create(
    #     self,
    #     subject: Token | SlotPrimitive,
    #     predicate: Token | SlotPrimitive,
    #     object: Token | SlotPrimitive,
    #     get_root=True,
    #     loc: Optional[SlotLoc] = None,
    # ):
    #     """Create new triple and add to TripleSet."""

    #     return self.get_or_create(subject, predicate, object, get_root=get_root, loc=loc)
