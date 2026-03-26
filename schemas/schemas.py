"""
Defines schemas/contracts to use inside of and between pipelines.
"""

from enum import Enum

import attr


class POSTag(Enum):
    """
    Part-of-speech tag

    Ref: https://universaldependencies.org/u/pos/
    """

    ADJ = "ADJ"
    """Adjective"""

    ADP = "ADP"
    """Adposition"""

    ADV = "ADV"
    """Adverb"""

    AUX = "AUX"
    """Auxiliary"""

    CCONJ = "CCONJ"
    """Coordinating conjunction"""

    DET = "DET"
    """Determiner"""

    INTJ = "INTJ"
    """Interjection"""

    NOUN = "NOUN"
    """Noun"""

    NUM = "NUM"
    """Numeral"""

    PART = "PART"
    """Particle"""

    PRON = "PRON"
    """Pronoun"""

    PROPN = "PROPN"
    """Proper noun"""

    PUNCT = "PUNCT"
    """Punctuation"""

    SCONJ = "SCONJ"
    """Subordinating conjunction"""

    SYM = "SYM"
    """Symbol"""

    VERB = "VERB"
    """Verb"""

    X = "X"
    """Other"""


@attr.s
class Token:
    """Represents an atomic unit of information in a sentence."""

    text: str = attr.ib()
    """Original text."""

    lemma: str = attr.ib()
    """If the token is a word, lemma is the gramatical root of the word."""

    pos: POSTag = attr.ib()
    """Part-of-speech tag for the token."""
