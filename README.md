# betterai-project

## Prerequisites

- UV: <https://docs.astral.sh/uv/getting-started/installation/>

## Setup

1. Sync submodules

   ```sh
   git submodule update --init --remote
   ```

2. Install dependencies

   ```sh
   uv sync
   
   python -m spacy download en_core_web_sm
   ```

## Helpful Commands

Installing new dependencies (ex: requests)

```sh
uv add requests
```

Run main.py

```sh
uv run main.py
```

## Evaluation

Run a simple baseline on a PubMedQA-style JSONL:

```sh
uv run evaluation/run_eval.py --dataset path/to/dev.jsonl --output runs/pqa_eval.jsonl --model heuristic --with_retrieval
```

JSONL format (fields per line):
- `id`: sample id
- `question`: text
- `answer` or `final_decision`: yes/no (case-insensitive)
- `context`: optional text

The runner computes Accuracy and Macro-F1, and can optionally attach retrieval summaries from the current KG.

Compute additional metrics on the predictions JSONL:

```sh
python -m evaluation.compute_metrics --pred runs/pqa_eval.jsonl --k 5 \
  --gold-entities-key gold_mesh --gold-pmids-key gold_pmids
```

Heuristic claim judging for faithfulness (optional):

```sh
python -m evaluation.judge_claims --input runs/pqa_eval.jsonl --output runs/pqa_eval_judged.jsonl --answer-key pred
python -m evaluation.compute_metrics --pred runs/pqa_eval_judged.jsonl --k 5
```

### Metrics Overview

- QA quality
  - Accuracy: fraction of correct yes/no answers.
  - Macro‑F1: average F1 over the "yes" and "no" classes (robust to imbalance).
- Retrieval quality
  - Coverage: fraction of samples with non‑empty `retrieval_sources`.
  - Precision@k / Recall@k: correctness and completeness among top‑k retrieved items.
  - MRR: how early the first correct item appears (higher is better).
  - NDCG: rank quality rewarding correct items near the top.
  - Note: to compute P@k/R@k/MRR/NDCG, include gold lists in your predictions JSONL, e.g. `gold_mesh` (MeSH IDs) and/or `gold_pmids` (PMIDs). Pass their keys via `--gold-entities-key` / `--gold-pmids-key`.
- Faithfulness (hallucination proxy)
  - Heuristic judge splits the answer into sentence‑like claims and checks word‑overlap against source titles/contents.
  - Outputs per‑claim verdicts: Supported or NEI (Not‑Enough‑Info).
  - Aggregates:
    - Hallucination Rate = (Contradicted + NEI) ÷ total claims (in heuristic, we only produce NEI/Supported).
    - Factual Precision = Supported ÷ total claims.

### Typical Flow

1. Run evaluation to produce predictions JSONL (optionally with `--with_retrieval`).
2. Compute QA + retrieval metrics with `evaluation.compute_metrics`.
3. (Optional) Run `evaluation.judge_claims` to annotate claims, then re‑run `evaluation.compute_metrics` to include faithfulness scores.


## Datasets Used

Below are the datasets used for evaluation and benchmarking hallucination reduction in RDF-grounded LMs.

**PubMedQA** - [Link](https://drive.google.com/drive/folders/1rOM_Y0FbmqsuJqjqcu7bntM1cB9QoKRn?usp=sharing)



---

### Notes

- `pubmedqa.jsonl` → Main evaluation benchmark (factual accuracy, hallucination rate). - Biomedical yes/no QA dataset with verified human labels (1k samples). 
- `pubmedqa_artificial.jsonl` → Optional training/calibration dataset. - Automatically labeled synthetic QA pairs (211k). Used for model calibration or pre-training.
- `pubmedqa_unlabeled.jsonl` → Retrieval stress-testing for RDF graph coverage. - Questions and contexts without gold labels (61k). Used for retrieval evaluation. 
- `medqa.jsonl` → Complex reasoning benchmark (clinical multi-choice).  
- `medhalt.jsonl` → Hallucination stress test for medical text generation.

---

