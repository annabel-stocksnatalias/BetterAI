from pathlib import Path
from typing import Dict, Optional

import joblib


def always_yes(_: Dict) -> str:
    return "yes"


NEG_Q_CUES = {
    " not ",
    " no ",
    " without ",
    " lack ",
    " none ",
    " absent ",
    " fail to ",
    " fails to ",
    " did not ",
    " does not ",
    " do not ",
    " is not ",
    " are not ",
}

NEG_CTX_CUES = {
    "not associated",
    "no association",
    "no significant association",
    "no significant difference",
    "no significant effect",
    "no effect",
    "no benefit",
    "no improvement",
    "no evidence",
    "did not improve",
    "did not reduce",
    "did not increase",
    "did not change",
    "was not superior",
    "were not superior",
    "not superior",
}

POS_CTX_CUES = {
    "significantly associated",
    "strongly associated",
    "significant association",
    "significant improvement",
    "significantly improved",
    "significantly reduced",
    "significantly decreased",
    "significant reduction",
    "increased risk",
    "associated with",
    "was superior",
    "were superior",
    "more effective",
    "effective in",
}


_ML_MODEL: Optional[dict] = None


def _load_ml_model() -> dict:
    """Lazy-load the trained TF-IDF + logistic regression model and threshold."""
    global _ML_MODEL
    if _ML_MODEL is None:
        model_path = Path("runs/models/pubmedqa_tfidf.joblib")
        if not model_path.exists():
            raise RuntimeError(
                f"Trained model not found at {model_path}. "
                "Run `uv run python -m evaluation.train_pubmedqa_baseline` first."
            )
        obj = joblib.load(model_path)
        # Backwards compatibility: older runs may have saved the raw pipeline
        if isinstance(obj, dict) and "model" in obj:
            _ML_MODEL = obj
        else:
            _ML_MODEL = {"model": obj, "threshold": 0.5}
    return _ML_MODEL


def heuristic_yesno(sample: Dict) -> str:
    """Simple heuristic for yes/no using question and context.

    Priority:
      1) Strong negation cues in context → 'no'
      2) Strong positive cues in context → 'yes'
      3) Negation cues in question text → 'no'
      4) Otherwise default to 'yes'
    """

    q = f" {sample.get('question', '').lower()} "
    ctx = f" {sample.get('context', '').lower()} "

    # 1) Context-level negation cues
    if any(phrase in ctx for phrase in NEG_CTX_CUES):
        return "no"

    # 2) Context-level positive cues
    if any(phrase in ctx for phrase in POS_CTX_CUES):
        return "yes"

    # 3) Question-level negation cues
    if any(c in q for c in NEG_Q_CUES):
        return "no"

    # 4) Fallback
    return "yes"


def ml_yesno(sample: Dict) -> str:
    """ML-based yes/no using TF-IDF + logistic regression trained on PubMedQA."""

    bundle = _load_ml_model()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    question = sample.get("question") or ""
    context = sample.get("context") or ""
    text = f"{question} [SEP] {context}".strip()

    # If for some reason we have no text, fall back to heuristic
    if not text:
        return heuristic_yesno(sample)

    # Use calibrated threshold on the positive ("yes") probability
    proba_yes = model.predict_proba([text])[0][1]
    return "yes" if proba_yes >= threshold else "no"


_HF_MODEL = None
_HF_TOKENIZER = None


def transformer_yesno(sample: Dict) -> str:
    """Transformer-based yes/no using a fine-tuned HF model."""
    global _HF_MODEL, _HF_TOKENIZER

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    if _HF_MODEL is None or _HF_TOKENIZER is None:
        model_dir = Path("runs/models/pubmedqa_transformer")
        if not model_dir.exists():
            raise RuntimeError(
                f"Transformer model not found at {model_dir}. "
                "Run `uv run python -m evaluation.train_pubmedqa_transformer` first."
            )
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
        _HF_MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _HF_MODEL.to(device)
        _HF_MODEL.eval()

    question = sample.get("question") or ""
    context = sample.get("context") or ""
    retrieval = sample.get("retrieval_summary") or ""

    parts = [f"Question: {question.strip()}", f"Context: {context.strip()}"]
    if retrieval:
        parts.append(f"Retrieval: {retrieval.strip()}")
    text = " [SEP] ".join(parts)

    if not text.strip():
        return heuristic_yesno(sample)

    import torch

    enc = _HF_TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    device = next(_HF_MODEL.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = _HF_MODEL(**enc)
        logits = outputs.logits
        proba = torch.softmax(logits, dim=-1)[0]
        proba_yes = float(proba[1])

    return "yes" if proba_yes >= 0.5 else "no"
