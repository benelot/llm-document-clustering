"""Load the cleaned incidents from naive/preprocess_incidents.py."""

import os
from pathlib import Path

import pandas as pd

CLEANED_FILE = os.getenv("CLEANED_FILE", "cleaned_incidents.json")


def load_cleaned(path=CLEANED_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run naive/preprocess_incidents.py first or set CLEANED_FILE."
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    else:
        df = pd.read_json(path, orient="records")
    df["consolidated_text"] = df["consolidated_text"].fillna("").astype(str)
    df = df[df["consolidated_text"].str.strip() != ""].reset_index(drop=True)
    df["incident_id"] = df["incident_id"].astype(str)
    print(f"Loaded {len(df)} non-empty incidents from {path}")
    return df


def truncate(text, max_chars):
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "..."
