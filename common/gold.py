"""
Gold standard: hand-made groups of document IDs per hypothesis category.

A person read the documents and grouped them. The clustering never sees
these groups; they are only used to evaluate the result at the end.

File format (gold/<format>.json):
  {"Category A": ["INC1", "INC2"], "Category B": ["INC3"]}
For CIRS the IDs are the document IDs from preprocess_cirs.py
(RespondentID, or RespondentID_<field> with --unit field).
"""

import json
from pathlib import Path


def load_gold(path):
    """Return {category: [ids]} or None if the file does not exist."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[gold] no gold-standard file at {path}, evaluation skipped")
        return None
    with open(path, encoding="utf-8") as f:
        gold = json.load(f)
    gold = {str(k): [str(i) for i in v] for k, v in gold.items()}
    print(f"[gold] {len(gold)} categories, {sum(len(v) for v in gold.values())} IDs from {path}")
    return gold
