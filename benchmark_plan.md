# Benchmarking Plan – Hallucination Reduction in Language Models via RDF Graphs & Jamba

## Objective
Evaluate and benchmark how RDF-grounded retrieval and Jamba (Mixture-of-Experts + RAG hybrid) architectures reduce hallucinations in large language models (LLMs) compared to standard baselines.

---

## Goals
1. Measure **hallucination reduction** achieved through RDF grounding and Jamba integration.
2. Benchmark **retrieval effectiveness**, **factual accuracy**, **latency**, and **coverage**.
3. Provide reproducible and transparent evaluation scripts, data, and metrics.

---

## Baseline Models

| Tier | Model | Type | Purpose |
|------|--------|------|---------|
| A | **LLaMA-3-8B-Instruct** | Vanilla LM | Measures raw hallucination tendency |
| B | **RAG-LLaMA-3-8B (Text)** | Text-retrieval RAG | Evaluates unstructured evidence |
| C | **RDF-RAG (SPARQL)** | Graph-grounded RAG | Measures effect of structured grounding |
| D | **Jamba-1.5 + RDF** | MoE + KG retrieval | Proposed advanced architecture |
| E | **GPT-3.5-Turbo / Human Gold** | Reference | Provides upper bound for factual accuracy |

---

## Dataset Preparation

### 1. Source Datasets
- **PubMedQA** or **MedQA**: medical Q&A dataset with verified answers.
- **Domain-specific RDF Graphs** (UMLS, Bio2RDF, custom KG).

### 2. Data Schema
```json
{
  "id": 1,
  "question": "What is the treatment for diabetes?",
  "gold": "Insulin therapy and lifestyle modification.",
  "triples": [
    ["Diabetes", "treated_by", "Insulin"],
    ["Diabetes", "treated_by", "Exercise"]
  ]
}
```

### 3. Size
Start with **100–500** questions for pilot; scale to **1000+** for final benchmark.

---

## ⚙️ Evaluation Pipeline Structure

```
evaluation/
│
├── data/
│   ├── prompts.jsonl
│   ├── gold.jsonl
│   └── retrieved_triples.jsonl
│
├── runs/
│   ├── llama3_baseline/outputs.jsonl
│   ├── rag_text/outputs.jsonl
│   ├── rdf_rag/outputs.jsonl
│   └── jamba_rdf/outputs.jsonl
│
├── scripts/
│   ├── make_claims.py
│   ├── nli_agreement.py
│   ├── retrieval_scores.py
│   ├── calibrate_risk.py
│   └── compute_metrics.py
│
└── results/
    ├── summary_metrics.csv
    ├── hallucination_examples.md
    └── plots/
```

---

## 🧮 Metrics & Definitions

| Metric | Description | Ideal Trend |
|---------|--------------|-------------|
| **Hallucination Rate (H)** | Fraction of unsupported / contradicted claims | ↓ Lower |
| **Factual Precision** | Supported claims ÷ total claims | ↑ Higher |
| **ROUGE-L / F1** | Overlap with reference answers | ↑ Higher |
| **Retrieval Precision@k** | % of correct triples retrieved | ↑ Higher |
| **Latency** | Average response time (seconds) | ↓ Lower |
| **ECE / Brier Score** | Calibration quality of confidence gate | ↓ Lower |
| **Coverage** | % of questions answered (not abstained) | Moderate |

---

## Evaluation Steps

### Step 1 — Generate Outputs
Run all models on the same dataset:
```bash
python run_model.py --model llama3-8b --input data/prompts.jsonl --output runs/llama3_baseline/outputs.jsonl
python run_model.py --model rdf_rag --use_rdf True --output runs/rdf_rag/outputs.jsonl
```

### Step 2 — Compare with Evidence
Use **triple matching** or **LLM-as-Judge**:
```text
"Given these RDF triples and the model's answer, is it factually correct? (Yes/No)"
```

### Step 3 — Compute Metrics
Aggregate per-sample results:
```bash
python scripts/compute_metrics.py --input runs/ --output results/summary_metrics.csv
```

### Step 4 — Visualize
Produce comparative bar charts:
- Hallucination Rate vs Model
- Latency vs Accuracy
- Example “Good vs Hallucinated” answers

---

## Example Result Table

| Model | RDF | H ↓ | F1 ↑ | Latency (s) ↓ | Notes |
|--------|-----|-----|-----|---------------|-------|
| LLaMA-3-8B | ❌ | 0.45 | 0.61 | 1.2 | Pure LM |
| RAG-Text | ❌ | 0.36 | 0.68 | 2.3 | Text retrieval |
| RDF-RAG | ✅ | 0.28 | 0.75 | 2.9 | Structured grounding |
| Jamba + RDF | ✅ | **0.22** | **0.79** | 3.1 | Best factual accuracy |

---

## Calibration & Risk Gating

To prevent hallucination:
1. Compute **risk score** = f(evidence agreement, retrieval quality, logit confidence, self-consistency).  
2. If risk > τ → abstain or re-retrieve.  
3. Tune τ on dev set to balance factuality vs coverage.

Evaluate with Expected Calibration Error (ECE) and Risk–Coverage curve.

---

## Tools & Libraries
- `transformers`, `vllm`, `faiss`, `networkx`
- `rdflib`, `SPARQLWrapper`
- `openai`, `anthropic` (for LLM-judge)
- `pandas`, `matplotlib`, `seaborn`
- `tqdm`, `jsonlines`

---

## Deliverables
1. **Evaluation scripts** (`.py` files under `scripts/`)
2. **Result CSVs** (`summary_metrics.csv`)
3. **Visualizations** (`plots/`)
4. **Benchmark report** summarizing key findings
5. **Appendix:** qualitative examples (correct vs. hallucinated answers)

---

## Example Summary Statement
> Across 500 medical questions, the baseline LLaMA-3-8B model exhibited a hallucination rate of 45%. Incorporating RDF-based retrieval reduced hallucination to 28%, while the Jamba + RDF model achieved 22% — a 36–50% reduction overall with minimal latency increase. These results demonstrate the effectiveness of structured RDF grounding for mitigating hallucinations in medical NLP systems.

---

