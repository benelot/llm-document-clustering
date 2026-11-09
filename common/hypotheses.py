"""
Hypothesis clusters: small hand-made groups of document IDs, each with a name.

A person read the documents and grouped them. The clustering never sees
these groups; they are only used to evaluate the result at the end.

File format (hypotheses/<format>.json):
  {"Warnmeldung": ["INC1", "INC2"], "Pumpen / Perfusor": ["INC3"]}
For CIRS the IDs are the document IDs from preprocess_cirs.py
(RespondentID, or RespondentID_<field> with --unit field).
"""

import json
from pathlib import Path


def load_hypotheses(path):
    """Return {hypothesis cluster name: [ids]} or None if the file does not exist."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[hypotheses] no hypotheses file at {path}, evaluation skipped")
        return None
    with open(path, encoding="utf-8") as f:
        hypotheses = json.load(f)
    hypotheses = {str(k): [str(i) for i in v] for k, v in hypotheses.items()}
    print(f"[hypotheses] {len(hypotheses)} hypothesis clusters, {sum(len(v) for v in hypotheses.values())} IDs from {path}")
    return hypotheses
