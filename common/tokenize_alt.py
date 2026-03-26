from __future__ import annotations

import spacy
from spacy.tokens import Doc

# Try SciSpaCy (medical) model first; fall back to en_core_web_sm
try:
    _NLP = spacy.load("en_core_sci_md")
    try:
        from scispacy.abbreviation import AbbreviationDetector

        if "abbreviation_detector" not in _NLP.pipe_names:
            _NLP.add_pipe("abbreviation_detector")
    except Exception:
        # If scispacy or the abbreviation detector isn't available, just skip it
        pass
except Exception:
    # Fallback: generic English model
    _NLP = spacy.load("en_core_web_sm")


def tokenize_text(text: str) -> Doc:
    """
    Given input text, return a spaCy Doc used for processing.

    This function is used by both:
      - Pipeline 1 (tokens_to_rdf)
      - Pipeline 2 (tokens_to_query)

    It returns a Doc (not a custom dataclass) to stay compatible
    with the original project structure.
    """
    return _NLP(text)

