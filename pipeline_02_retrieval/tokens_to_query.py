from __future__ import annotations

import re
from textwrap import dedent

from spacy.tokens import Doc

# Use the same prefix as the ingestion pipeline to avoid mismatches
REL = "http://example.org/rel/"

# Mentions we never want to treat as the subject
_STOP_MENTIONS = {
    "which enzyme",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "enzyme",
    "drug",
    "medicine",
    # Additional generic/question words we never want as subject mentions
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "will",
    "shall",
}

# --- Intent detection patterns (your logic) -----------------------------------

_INTENT_PATTERNS = [
    (
        "mechanism_of_action",
        re.compile(
            r"\bmechanism of action\b|\bacts by\b|\binhibit(s|ing)?\b|\bstimulate(s|ing)?\b",
            re.I,
        ),
    ),
    (
        "indication",
        re.compile(
            r"\bindication(s)?\b|\bused for\b|\btreat(s|ment)? of\b",
            re.I,
        ),
    ),
    (
        "contraindication",
        re.compile(
            r"\bcontraindication(s)?\b|\bcontraindicated\b|\bavoid in\b",
            re.I,
        ),
    ),
    (
        "adverse_effect",
        re.compile(
            r"\bside effect(s)?\b|\badverse\b|\btoxicit(y|ies)\b",
            re.I,
        ),
    ),
    (
        "dose",
        re.compile(
            r"\bdose|dosage|dosing\b",
            re.I,
        ),
    ),
    (
        "drug_target",
        re.compile(
            r"\btarget(s)?\b|\bbinds?\b|\breceptor\b|\benzyme\b",
            re.I,
        ),
    ),
]


def _detect_intent(text: str) -> str | None:
    """Infer high-level intent (what relation is being asked about) from raw text."""
    for name, pat in _INTENT_PATTERNS:
        if pat.search(text):
            return name
    # Very rough fallback: "what ..." questions → assume mechanism_of_action
    if re.match(r"^\s*what\b", text.strip(), re.I):
        return "mechanism_of_action"
    return None


# --- Mention extraction (your improved heuristic) -----------------------------


def _extract_mentions(doc: Doc) -> list[str]:
    """
    Extract candidate 'mentions' (drug names, enzymes, etc.) from a Doc.

    Strategy:
      1. Prefer named entities (doc.ents) if available.
      2. Add noun chunks.
      3. Add capitalized / chemical-ish single tokens.
      4. De-duplicate (case-insensitive), keep track of first-seen position.
      5. Sort: longer first, tie-break by original position.
    """
    cands_with_pos: list[tuple[str, int]] = []
    seen_lower: set[str] = set()

    def _add(s: str, pos: int):
        s = s.strip()
        # Ignore very short fragments; they tend to be auxiliaries like "Do", "Is".
        if not (3 <= len(s) <= 80):
            return
        k = s.lower()
        if k in seen_lower:
            return
        seen_lower.add(k)
        cands_with_pos.append((s, pos))

    pos = 0

    # 1) named entities, if available
    if getattr(doc, "ents", None):
        for ent in doc.ents:
            _add(ent.text, pos)
            pos += 1

    # 2) noun chunks
    for span in getattr(doc, "noun_chunks", []):
        _add(span.text, pos)
        pos += 1

    # 3) capitalized / chemical-ish single tokens
    for t in doc:
        if re.search(r"[A-Za-z0-9α-ωΑ-Ω\-]+", t.text) and (
            t.shape_ in {"Xxxx", "XX", "XXX", "Xx"} or "-" in t.text
        ):
            _add(t.text, pos)
            pos += 1

    # 4) sort: longer first, tie-break by original position
    cands_with_pos.sort(key=lambda p: (-len(p[0]), p[1]))

    # 5) return just the strings
    return [s for s, _ in cands_with_pos]


# --- Subject / label binding helpers -----------------------------------------


def _sanitize_mention(m: str) -> str:
    # Keep only harmless chars inside a quoted string
    m = re.sub(r'["\\]', " ", m)
    return " ".join(m.split()).strip()


def _pick_best_mention(mentions: list[str]) -> str | None:
    """
    From a list of mention strings, pick the best candidate for the subject.
    """
    # 1) prefer good-looking biomedical strings with uppercase/digits/hyphens
    for m in mentions:
        m_clean = m.strip()
        if not m_clean:
            continue
        low = m_clean.lower()
        if low in _STOP_MENTIONS:
            continue
        if re.search(r"[A-Z0-9\-]", m_clean):
            return m_clean

    # 2) fallback: first non-stop mention
    for m in mentions:
        if m.strip().lower() not in _STOP_MENTIONS:
            return m.strip()

    return None


def _wrap_prefixes(q: str) -> str:
    return dedent(
        f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rel: <{REL}>
        {q.strip()}
        """
    ).strip()


def _extract_kb_ids(doc: Doc) -> list[str]:
    """
    Collect KB IDs from entities if available (e.g., SciSpaCy linker).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for ent in getattr(doc, "ents", []) or []:
        kb_id = getattr(ent, "kb_id_", "") or ""
        if kb_id:
            k = kb_id.strip()
            if k and k not in seen:
                seen.add(k)
                ids.append(k)
        # scispacy_linker exposes ent._.kb_ents as list of (id, score)
        if hasattr(ent._, "kb_ents"):
            for kb, _score in getattr(ent._, "kb_ents", []) or []:
                if kb and kb not in seen:
                    seen.add(kb)
                    ids.append(kb)
    return ids


def _subject_binding_inline_filter(mention: str, kb_ids: list[str] | None = None) -> str:
    """
    Bind ?subj by label, and optionally by known KB IDs (e.g., MeSH codes).
    """
    m = _sanitize_mention(mention)
    blocks = [
        dedent(
            f"""
            {{
              ?subj rdfs:label ?lbl .
              FILTER( CONTAINS(LCASE(STR(?lbl)), LCASE("{m}")) )
            }}
            """
        ).strip()
    ]

    kb_ids = kb_ids or []
    for kb in kb_ids:
        kb_clean = _sanitize_mention(kb)
        if not kb_clean:
            continue
        blocks.append(
            dedent(
                f"""
                {{
                  ?subj rel:has_mesh_id ?mid .
                  FILTER( LCASE(STR(?mid)) = LCASE("{kb_clean}") )
                }}
                """
            ).strip()
        )

    return "\nUNION\n".join(blocks)


# --- Main: Doc → SPARQL query -------------------------------------------------


def tokens_to_query(tokens: Doc) -> str:
    """
    Convert a spaCy Doc (tokenized question) into a SPARQL query string.

    This uses:
      - intent detection (mechanism_of_action, indication, etc.)
      - mention extraction (drug/target/etc. names)
      - label-based subject binding in the RDF graph
    """

    text = tokens.text

    # 1) Extract candidate mentions from the Doc
    mentions = _extract_mentions(tokens)

    # 2) Expand mentions using abbreviations, if SciSpaCy abbreviation detector is present
    abbrev_map: dict[str, str] = {}
    if hasattr(tokens._, "abbreviations"):
        for ab in getattr(tokens._, "abbreviations", []):
            abbrev_map[str(ab)] = str(ab._.long_form)

    expanded = [abbrev_map.get(m, m) for m in mentions]
    # deduplicate while preserving order
    mentions = list(dict.fromkeys(expanded))

    # 3) Detect intent from the raw text
    intent = _detect_intent(text)

    # 3.5) Collect KB IDs from entities (if linker present)
    kb_ids = _extract_kb_ids(tokens)

    # 4) Choose the best subject mention
    chosen = _pick_best_mention(mentions)
    if not chosen:
        # Harmless always-empty query if we couldn't find a subject
        return _wrap_prefixes("SELECT ?answer WHERE { VALUES ?answer { } }")

    subj_bind = _subject_binding_inline_filter(chosen, kb_ids=kb_ids)

    # 5) Build a filtered block (intent-aware) and a general fallback block.
    #    This avoids empty results when intent-specific predicates are missing.
    intent_filters = []
    if intent:
        # Use simple keyword filters over predicate IRIs to stay schema-agnostic.
        intent_kw = intent.replace("_", " ")
        for kw in intent_kw.split():
            kw = kw.strip()
            if not kw:
                continue
            intent_filters.append(f'CONTAINS(LCASE(STR(?predicate)), "{kw.lower()}")')

    intent_filter_expr = " || ".join(intent_filters) if intent_filters else ""

    intent_block = dedent(
        f"""
        {{
          {subj_bind}
          ?subj ?predicate ?object .
          OPTIONAL {{ ?object rdfs:label ?objectLabel . }}
          {"FILTER(" + intent_filter_expr + ")" if intent_filter_expr else ""}
        }}
        """
    ).strip()

    fallback_block = dedent(
        f"""
        {{
          {subj_bind}
          ?subj ?predicate ?object .
          OPTIONAL {{ ?object rdfs:label ?objectLabel . }}
        }}
        """
    ).strip()

    body = dedent(
        f"""
        SELECT ?predicate ?object ?objectLabel WHERE {{
          {intent_block}
          UNION
          {fallback_block}
        }}
        LIMIT 100
        """
    )
    return _wrap_prefixes(body)
