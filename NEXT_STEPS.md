# Hallucination Reduction Plan

## Objectives
- Make retrieval hit the graph reliably and return scored evidence.
- Force generation to ground answers in cited evidence or explicitly refuse.
- Add verification and metrics so regressions are caught automatically.

## Current Gaps (why hallucination happens)
- SPARQL predicates/prefixes don’t match ingested data; many queries return empty.
- Subjects often lack `rdfs:label`, so label binding fails.
- Biomedical entities are not linked to canonical IDs; mention detection is generic.
- No fallback retriever or scoring; empty/low-quality evidence leads to guessing.
- Generation layer doesn’t exist; no citation or refusal logic.
- Claim verification is minimal and offline-only; tests/metrics don’t guard faithfulness.

## Work Plan (ordered)
1) **Schema Alignment**
   - Pick one prefix (e.g., `http://example.org/rel/`) and map predicates used in `tokens_to_query.py` to those written by ingestion (`has_mesh_major`, `has_pubmedqa_question/context/answer`, `has_title`, `has_abstract`, etc.).
   - Ensure every subject gets an `rdfs:label` on ingest (`Database.apply_tripleset`/`apply_json`).
   - Add a tiny fixture graph + question to tests that asserts the SPARQL returns non-empty rows.

2) **Entity Linking & Mention Binding**
   - Switch `common/tokenize.py` to use SciSpaCy + UMLS/MeSH linker (keep a lightweight fallback).
   - In `tokens_to_query`, prefer ID-based binding (PMID/MeSH/DrugBank) with a text `CONTAINS` fallback; expand abbreviations via scispacy.
   - Keep/extend regex intent detection; add a “unknown intent → outbound edges” fallback.

3) **Robust Retrieval with Scoring**
   - Add lexical fallback (BM25/TF-IDF) over `has_title`/`has_abstract`/textbooks when SPARQL is empty; optionally add embedding search.
   - Merge results, dedupe, and carry a `score` field in `DocSource`; drop or warn when scores are low.

4) **Grounded Generation (RAG)**
   - Add a generator that takes `DocSource` list and produces answers with inline citations (e.g., `[1]` mapped to source ids/titles).
   - Implement refusal logic: if no sources or max score below threshold, answer “Not enough evidence to answer.”
   - Wire this into evaluation so models can consume retrieval summaries/sources.

5) **Claim Verification**
   - Upgrade `evaluation/judge_claims.py` to use a better verifier (entailment model or LLM) against retrieved snippets; support Supported/NEI/Contradicted.
   - Optionally run verifier post-generation to drop or flag unsupported claims before returning.

6) **Patient/Context Filtering (optional but useful)**
   - Implement `pipeline_02_retrieval/patient_context.py` filters (age/sex/conditions) to narrow evidence and avoid irrelevant hits.

7) **Tests & Metrics**
   - Add E2E pytest: seed a tiny graph; ask a question; assert retrieval non-empty → answer has citations; retrieval empty → explicit refusal.
   - Add retrieval coverage + P@k/MRR/NDGC to CI via `evaluation.compute_metrics`.
   - Add hallucination tracking via `evaluation.judge_claims` + `compute_metrics` to monitor hallucination rate/factual precision.

## How to Measure Improvement (manual checks)
- Run retrieval + QA: `uv run python -m evaluation.run_eval --dataset data/pubmedqa/pubmedqa.jsonl --output runs/pqa_eval.jsonl --model heuristic --with_retrieval`
- Compute metrics: `python -m evaluation.compute_metrics --pred runs/pqa_eval.jsonl --k 5` (watch Retrieval Coverage/P@k/MRR/NDCG).
- Hallucination proxy: `python -m evaluation.judge_claims --input runs/pqa_eval.jsonl --output runs/pqa_eval_judged.jsonl --answer-key pred` then `python -m evaluation.compute_metrics --pred runs/pqa_eval_judged.jsonl --k 5` (watch Hallucination/FactualPrecision).
