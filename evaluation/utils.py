import json
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_yes_no(label: str) -> str:
    """Normalize labels to 'yes' or 'no'."""
    s = (label or "").strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    # PubMedQA sometimes uses 'yes'/'no'/'maybe'; map maybe to 'no' by default
    if s == "maybe":
        return "no"
    return s

