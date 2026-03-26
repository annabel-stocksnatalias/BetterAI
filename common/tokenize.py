import contractions
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token


def _load_nlp():
    """
    Try to load a biomedical model if available; otherwise fall back to en_core_web_sm.
    """
    # SciSpaCy small model name; adjust if a different biomedical model is installed.
    candidates = ["en_core_sci_sm", "en_core_web_sm"]
    for name in candidates:
        try:
            nlp = spacy.load(name)
            # Best-effort: add abbreviation detector and linker if present
            try:
                if "abbreviation_detector" not in nlp.pipe_names:
                    nlp.add_pipe("abbreviation_detector")
            except Exception:
                pass
            try:
                if "scispacy_linker" not in nlp.pipe_names:
                    nlp.add_pipe("scispacy_linker")
            except Exception:
                # Not fatal if linker is unavailable
                pass
            return nlp
        except Exception:
            continue
    # Last resort: blank English pipeline
    return spacy.blank("en")


# Load the spaCy model once at import time and reuse it.
_NLP = _load_nlp()


def tokenize_text(text: str, enable_bert: bool = False) -> Doc:
    """Given input text, return tokenized version used for processing."""

    nlp = _NLP

    if enable_bert:
        # Best-effort transformer setup; safe to skip if unavailable
        try:
            if "transformer" not in nlp.pipe_names:
                nlp.add_pipe(
                    "transformer",
                    config={"model": {"name": "dmis-lab/biobert-v1.1"}},
                )
        except Exception:
            # If transformers or the specific model isn't available, fall back gracefully.
            pass

    text = contractions.fix(text)

    # nlp.add_pipe("experimental_coref")
    # nlp.initialize()

    doc = nlp(text)

    # Add custom attribute "noun_chunk" to tokens
    Token.set_extension("noun_chunk", default=None, force=True)

    for noun in doc.noun_chunks:
        noun.root._.noun_chunk = noun

    return doc
