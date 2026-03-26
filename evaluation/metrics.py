from collections import Counter
from typing import Iterable, List, Tuple


def accuracy_and_macro_f1(golds: Iterable[str], preds: Iterable[str]) -> Tuple[float, float]:
    g = list(golds)
    p = list(preds)
    assert len(g) == len(p)

    total = len(g)
    correct = sum(1 for i in range(total) if g[i] == p[i])
    acc = correct / total if total else 0.0

    labels = sorted(set(g) | set(p))
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for i in range(total) if g[i] == lab and p[i] == lab)
        fp = sum(1 for i in range(total) if g[i] != lab and p[i] == lab)
        fn = sum(1 for i in range(total) if g[i] == lab and p[i] != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return acc, macro_f1

