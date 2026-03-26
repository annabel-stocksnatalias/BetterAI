import ast
import csv
import json
from pathlib import Path

from database.database import Database
from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDFS


NS = Namespace("http://example.org/node/")


def slug(text: str) -> str:
    """Simple slugifier to mirror Database.apply_json URI scheme."""
    text = str(text or "").strip().lower()
    import re as _re

    text = _re.sub(r"\s+", "_", text)
    text = _re.sub(r"[^a-z0-9_\-]", "", text)
    return text or "unnamed"


def ingest_mesh_csv(db: Database, csv_path: Path, max_rows: int | None = 1000) -> int:
    """
    Ingest a MeSH-annotated PubMed CSV into the RDF graph.

    Expects columns (based on the Kaggle MeSH dataset):
      - Title
      - abstractText
      - meshMajor        (e.g. "['DNA Probes, HPV', 'DNA, Viral', ...]")
      - pmid
      - meshid           (e.g. "[['D13.444.600.223.555', ...], ...]")

    Only a subset of rows is ingested by default (max_rows) to keep things fast.
    Returns the number of triples added.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"MeSH CSV not found at: {csv_path}")

    triples_batch: list[dict] = []
    total_triples = 0

    # Pre-count rows for simple progress reporting (excluding header).
    try:
        with csv_path.open(newline="", encoding="utf-8") as f_count:
            total_rows = max(sum(1 for _ in f_count) - 1, 0)
    except OSError:
        total_rows = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            pmid = (row.get("pmid") or "").strip()
            if not pmid:
                continue

            title = (row.get("Title") or "").strip()
            abstract = (row.get("abstractText") or "").strip()
            mesh_major_raw = row.get("meshMajor") or ""
            mesh_id_raw = row.get("meshid") or ""

            # Basic article metadata
            if title:
                triples_batch.append({"s": pmid, "p": "has_title", "o": title})
                total_triples += 1

                # Also expose title as an rdfs:label so retrieval can bind by label
                subj_uri = URIRef(NS + slug(pmid))
                db.graph.add((subj_uri, RDFS.label, Literal(title)))
            if abstract:
                triples_batch.append({"s": pmid, "p": "has_abstract", "o": abstract})
                total_triples += 1

            # Parse meshMajor: list of human-readable MeSH headings
            try:
                major_terms = ast.literal_eval(mesh_major_raw)
                if not isinstance(major_terms, list):
                    major_terms = []
            except (SyntaxError, ValueError):
                major_terms = []

            for term in major_terms:
                term_text = str(term).strip().strip("'\"")
                if not term_text:
                    continue
                triples_batch.append({"s": pmid, "p": "has_mesh_major", "o": term_text})
                total_triples += 1

            # Parse meshid: nested list of MeSH tree identifiers
            try:
                mesh_ids_nested = ast.literal_eval(mesh_id_raw)
                if not isinstance(mesh_ids_nested, list):
                    mesh_ids_nested = []
            except (SyntaxError, ValueError):
                mesh_ids_nested = []

            for group in mesh_ids_nested:
                if not isinstance(group, list):
                    continue
                for mesh_id in group:
                    code = str(mesh_id).strip()
                    if not code:
                        continue
                    triples_batch.append({"s": pmid, "p": "has_mesh_id", "o": code})
                    total_triples += 1

            # Flush in batches to avoid huge in-memory payloads
            if len(triples_batch) >= 5000:
                db.apply_json(triples_batch)
                triples_batch.clear()
                print(f"[MeSH] Flushed batch at row {row_idx:,}; total triples so far: {total_triples:,}")

            # Lightweight progress indicator.
            if row_idx % 5000 == 0 or (total_rows and row_idx == total_rows):
                if total_rows:
                    pct = (row_idx / total_rows) * 100
                    print(f"[MeSH] Progress: {row_idx:,}/{total_rows:,} rows ({pct:.1f}%).")
                else:
                    print(f"[MeSH] Processed {row_idx:,} rows...")

            if max_rows is not None and row_idx >= max_rows:
                break

    if triples_batch:
        db.apply_json(triples_batch)

    return total_triples


def ingest_pubmedqa_jsonl(db: Database, jsonl_path: Path, max_rows: int | None = 1000) -> int:
    """
    Ingest PubMedQA JSONL into the RDF graph.

    Each line is expected to contain at least:
      - id
      - question
      - context (abstract)
      - gold (yes/no), when available

    We attach question/context/answer triples to the article node keyed by PMID.
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"PubMedQA JSONL not found at: {jsonl_path}")

    triples_batch: list[dict] = []
    total_triples = 0

    # Pre-count lines for progress reporting.
    try:
        with jsonl_path.open(encoding="utf-8") as f_count:
            total_lines = sum(1 for _ in f_count)
    except OSError:
        total_lines = 0

    with jsonl_path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            pmid = str(row.get("id") or "").strip()
            if not pmid:
                continue

            question = (row.get("question") or "").strip()
            context = (row.get("context") or "").strip()
            gold = (row.get("gold") or "").strip()

            if question:
                triples_batch.append({"s": pmid, "p": "has_pubmedqa_question", "o": question})
                total_triples += 1
                subj_uri = URIRef(NS + slug(pmid))
                db.graph.add((subj_uri, RDFS.label, Literal(question)))

            if context:
                triples_batch.append({"s": pmid, "p": "has_pubmedqa_context", "o": context})
                total_triples += 1

            if gold:
                triples_batch.append({"s": pmid, "p": "has_pubmedqa_answer", "o": gold})
                total_triples += 1

            if len(triples_batch) >= 5000:
                db.apply_json(triples_batch)
                triples_batch.clear()
                print(f"[PubMedQA] Flushed batch at row {row_idx:,}; total triples so far: {total_triples:,}")
            # Progress indicator.
            if row_idx % 5000 == 0 or (total_lines and row_idx == total_lines):
                if total_lines:
                    pct = (row_idx / total_lines) * 100
                    print(f"[PubMedQA] Progress: {row_idx:,}/{total_lines:,} lines ({pct:.1f}%).")
                else:
                    print(f"[PubMedQA] Processed {row_idx:,} lines...")

            if max_rows is not None and row_idx >= max_rows:
                break

    if triples_batch:
        db.apply_json(triples_batch)

    return total_triples


def ingest_medqa_usmle(
    db: Database,
    base_dir: Path,
    splits: list[str] | None = None,
    max_rows_per_split: int | None = 1000,
) -> int:
    """
    Ingest MedQA-USMLE question JSONL files into the RDF graph.

    For each question we create a node and attach:
      - has_question
      - has_answer (gold)
      - has_option (for each option)
      - has_meta (meta_info)
    """

    if splits is None:
        splits = ["train", "dev", "test"]

    total_triples = 0
    triples_batch: list[dict] = []

    for split in splits:
        jsonl_path = base_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue

        # Pre-count lines for progress reporting per split.
        try:
            with jsonl_path.open(encoding="utf-8") as f_count:
                total_lines = sum(1 for _ in f_count)
        except OSError:
            total_lines = 0

        with jsonl_path.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q_text = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                options = row.get("options") or {}
                meta = (row.get("meta_info") or "").strip()

                if not q_text:
                    continue

                qid = f"medqa_us_{split}_{row_idx}"

                triples_batch.append({"s": qid, "p": "has_question", "o": q_text})
                total_triples += 1

                if answer:
                    triples_batch.append({"s": qid, "p": "has_answer", "o": answer})
                    total_triples += 1

                if isinstance(options, dict):
                    for key, text in options.items():
                        opt_text = str(text).strip()
                        if not opt_text:
                            continue
                        label = f"{key}. {opt_text}"
                        triples_batch.append({"s": qid, "p": "has_option", "o": label})
                        total_triples += 1

                if meta:
                    triples_batch.append({"s": qid, "p": "has_meta", "o": meta})
                    total_triples += 1

                # Expose question as rdfs:label for subject node
                subj_uri = URIRef(NS + slug(qid))
                db.graph.add((subj_uri, RDFS.label, Literal(q_text)))

                if len(triples_batch) >= 5000:
                    db.apply_json(triples_batch)
                    triples_batch.clear()
                    print(
                        f"[MedQA-USMLE] Flushed batch at split {split}, row {row_idx:,}; "
                        f"total triples so far: {total_triples:,}"
                    )
                # Progress indicator.
                if row_idx % 5000 == 0 or (total_lines and row_idx == total_lines):
                    if total_lines:
                        pct = (row_idx / total_lines) * 100
                        print(
                            f"[MedQA-USMLE] Progress ({split}): "
                            f"{row_idx:,}/{total_lines:,} questions ({pct:.1f}%)."
                        )
                    else:
                        print(f"[MedQA-USMLE] Processed {row_idx:,} questions from split {split}...")

                if max_rows_per_split is not None and row_idx >= max_rows_per_split:
                    break

    if triples_batch:
        db.apply_json(triples_batch)

    return total_triples


def ingest_medqa_textbooks_en(
    db: Database,
    textbooks_dir: Path,
    max_chars_per_book: int | None = None,
) -> int:
    """
    Ingest MedQA-USMLE English textbooks as RDF triples.

    Each textbook file becomes a subject node with:
      - has_textbook_title
      - has_textbook_content  (optionally truncated to max_chars_per_book)
    """

    if not textbooks_dir.exists():
        raise FileNotFoundError(f"MedQA textbooks directory not found at: {textbooks_dir}")

    total_triples = 0
    triples_batch: list[dict] = []

    # Count textbooks for progress reporting.
    paths = sorted(textbooks_dir.glob("*.txt"))
    total_books = len(paths)

    for idx, path in enumerate(paths, start=1):
        book_id = path.stem  # e.g., "Anatomy_Gray"
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if max_chars_per_book is not None and len(text) > max_chars_per_book:
            text = text[:max_chars_per_book]

        title = book_id.replace("_", " ")

        triples_batch.append({"s": book_id, "p": "has_textbook_title", "o": title})
        total_triples += 1

        triples_batch.append({"s": book_id, "p": "has_textbook_content", "o": text})
        total_triples += 1

        # Expose textbook title as rdfs:label
        subj_uri = URIRef(NS + slug(book_id))
        db.graph.add((subj_uri, RDFS.label, Literal(title)))

        if len(triples_batch) >= 1000:
            db.apply_json(triples_batch)
            triples_batch.clear()
            print(
                f"[MedQA-Textbooks] Flushed batch after book {book_id}; "
                f"total triples so far: {total_triples:,}"
            )
        # Progress indicator.
        if total_books:
            pct = (idx / total_books) * 100
            print(
                f"[MedQA-Textbooks] Progress: {idx:,}/{total_books:,} books ({pct:.1f}%)."
            )

    if triples_batch:
        db.apply_json(triples_batch)

    return total_triples


def main():
    import os

    db = Database()

    # Optional environment knobs to limit graph size for faster experiments.
    mesh_max_rows_env = os.getenv("BETTERAI_MESH_MAX_ROWS")
    pubmedqa_max_rows_env = os.getenv("BETTERAI_PUBMEDQA_MAX_ROWS")
    medqa_max_rows_env = os.getenv("BETTERAI_MEDQA_MAX_ROWS_PER_SPLIT")
    textbooks_max_chars_env = os.getenv("BETTERAI_TEXTBOOK_MAX_CHARS")

    mesh_max_rows = int(mesh_max_rows_env) if mesh_max_rows_env else None
    pubmedqa_max_rows = int(pubmedqa_max_rows_env) if pubmedqa_max_rows_env else None
    medqa_max_rows = int(medqa_max_rows_env) if medqa_max_rows_env else None
    textbooks_max_chars = int(textbooks_max_chars_env) if textbooks_max_chars_env else None

    # --- 1) Ingest MeSH dataset into the RDF graph ---------------------------
    mesh_csv = Path("data/MeSH") / "PubMed Multi Label Text Classification Dataset Processed.csv"

    print(f"Ingesting MeSH data from: {mesh_csv} (max_rows={mesh_max_rows})")
    triples_added_mesh = ingest_mesh_csv(db=db, csv_path=mesh_csv, max_rows=mesh_max_rows)
    print(f"Added approximately {triples_added_mesh} MeSH triples to the RDF graph.")
    print(f"Graph now contains {len(db.graph)} triples (including any existing data).")

    # --- 1.5) Ingest PubMedQA JSONL into the RDF graph ----------------------
    pubmedqa_jsonl = Path("data/pubmedqa") / "pubmedqa.jsonl"
    print(f"Ingesting PubMedQA data from: {pubmedqa_jsonl} (max_rows={pubmedqa_max_rows})")
    triples_added_pubmedqa = ingest_pubmedqa_jsonl(
        db=db,
        jsonl_path=pubmedqa_jsonl,
        max_rows=pubmedqa_max_rows,
    )
    print(f"Added approximately {triples_added_pubmedqa} PubMedQA triples to the RDF graph.")
    print(f"Graph now contains {len(db.graph)} triples (including any existing data).")

    # --- 2) Ingest MedQA-USMLE questions into the RDF graph -----------------
    medqa_base = Path("data/USMLE/MedQA-USMLE/questions/US")
    print(f"Ingesting MedQA-USMLE data from: {medqa_base} (max_rows_per_split={medqa_max_rows})")
    triples_added_medqa = ingest_medqa_usmle(
        db=db,
        base_dir=medqa_base,
        max_rows_per_split=medqa_max_rows,
    )
    print(f"Added approximately {triples_added_medqa} MedQA-USMLE triples to the RDF graph.")
    print(f"Graph now contains {len(db.graph)} triples (including any existing data).")

    # --- 3) Ingest MedQA-USMLE English textbooks into the RDF graph ---------
    medqa_textbooks_en = Path("data/USMLE/MedQA-USMLE/textbooks/en")
    print(f"Ingesting MedQA-USMLE textbooks from: {medqa_textbooks_en} (max_chars_per_book={textbooks_max_chars})")
    triples_added_textbooks = ingest_medqa_textbooks_en(
        db=db,
        textbooks_dir=medqa_textbooks_en,
        max_chars_per_book=textbooks_max_chars,
    )
    print(f"Added approximately {triples_added_textbooks} MedQA textbook triples to the RDF graph.")
    print(f"Graph now contains {len(db.graph)} triples (including any existing data).")

    # --- 4) (Optional) Run the existing toy pipeline on a small example ------
    # This step depends on extra NLP / transformer libraries (spaCy, transformers, etc.).
    # If they're not installed, we skip the toy example but keep the graph.
    try:
        from pipeline_01_processing.pipeline import run_pipeline as run_pipeline_1
        from pipeline_02_retrieval.pipeline import run_pipeline as run_pipeline_2
    except ModuleNotFoundError as e:
        print(f"[Toy pipeline] Skipping example retrieval pipeline (missing dependency: {e}).")
        return

    source_text = (
        "High blood pressure is a common condition that affects the body's arteries. "
        "It's also called hypertension. If you have high blood pressure, "
        "the force of the blood pushing against the artery walls is consistently too high. "
        "The heart has to work harder to pump blood."
    )

    query_text = "What is hypertension?"

    # Convert source text to RDF structure (toy example)
    out_1 = run_pipeline_1(text=source_text)
    print("Toy RDF input structure from pipeline 1:", out_1)
    # Apply RDF structure from the toy pipeline to the database
    db.apply_tripleset(out_1)

    # Execute retrieval pipeline against the populated graph
    out_2 = run_pipeline_2(db=db, text=query_text)
    print("Answer output:", out_2)


if __name__ == "__main__":
    main()
