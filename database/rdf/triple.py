from typing import Optional, Union

import attrs
from rdflib import Literal, Namespace, URIRef

# Slot = Union[str | "Triple"]

SlotPrimitive = Union[str, int, "Triple"]

SlotLoc = tuple[str, int, int]
NS = Namespace("http://localhost:8000/source/")
REL = Namespace("http://example.com/rel/")


class Slot:
    """Represents a single value in a triple."""

    value: SlotPrimitive
    loc: Optional[SlotLoc] = None

    def __init__(self, value: SlotPrimitive, loc: Optional[SlotLoc] = None):
        self.value = value
        self.loc = loc

    def __str__(self):
        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, int):
            return self.value
        else:
            return f"({self.value})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def __eq__(self, value):
        if value is None:
            return True
        elif isinstance(value, Slot):
            return value.value == self.value
        else:
            return value == self.value

    def to_rdf(self):
        if self.loc is not None:
            source, start, end = self.loc
            return URIRef(NS + source + f"?start={start}&end={end}")
        else:
            return Literal(self.value)


class Node(Slot):
    """Represents the subject or object of a triple."""

    pass


class Pred(Slot):
    """Represents the preposition part of a triple."""

    pass


SlotLike = Union[Slot, SlotPrimitive]


@attrs.define
class Triple:
    """
    Represents an RDF triple of the for <subject, predicate, object>
    """

    subject: Node
    """Position 1 of triple, primary party of the triple statement."""

    predicate: Pred
    """Position 2 of triple, usually a verb, connects the subject and object."""

    object: Node
    """Position 3 of triple, entity that is related to the subject."""

    def as_tuple(self):
        return (self.subject, self.predicate, self.object)

    def __repr__(self):
        return f"<subject: {self.subject}, predicate: {self.predicate}, object: {self.object}>"

    def __eq__(self, value):
        subject = None
        predicate = None
        object_ = None

        if isinstance(value, tuple):
            if not len(value) == 3:
                return False

            subject, predicate, object_ = value
        elif isinstance(value, Triple):
            subject, predicate, object_ = value.as_tuple()
        else:
            return False

        s_match = subject == self.subject if subject is not None else True
        p_match = predicate == self.predicate if predicate is not None else True
        o_match = object_ == self.object if object_ is not None else True

        return s_match and p_match and o_match
