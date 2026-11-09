#!/usr/bin/env python3
"""
Preprocess CIRS / EHR survey exports (Excel or CSV) into the same format as
preprocess_incidents.py, so that every pipeline in this repo can read it.

Expected columns (names are matched case- and whitespace-insensitively):
  metadata:  RespondentID, MeasurementName, Clinical Background,
             Clinical Background Raw, Average Experience Rating,
             Your Specialty, Locations of Use, EMR, PrimaryWork Location
  free text: Fix 1, Fix 2, Fix 3, Most valuable improvements,
             CommentsGeneral, Disagree IT Seeking to Improve EHR Comments,
             IT Department Improve EHR Comments, Useful Training Comment,
             AI greatest potential not currently used,
             Hardware issues description

Unit of clustering (CIRS_UNIT):
  field       (default) one document per non-empty free-text answer.
              ID = <RespondentID>_<field>, e.g. 1234_fix1. One respondent
              writes about several topics, so this gives cleaner topics.
  respondent  one document per respondent, all answers joined.

Other settings:
  CIRS_INCLUDE_QUESTION  1 = prefix each answer with its question in the text
                         (more context for short answers, but answers to the
                         same question then lean towards each other). Default 0.
  CIRS_MIN_CHARS         answers shorter than this are dropped (default 3).

Usage:
  python preprocess_cirs.py [input.xlsx|input.csv] [output.csv]
  default: cirs.xlsx -> cleaned_cirs.csv + cleaned_cirs.json

Then run any pipeline with:
  export CLEANED_FILE=cleaned_cirs.json DATA_PROFILE=cirs
"""

import os
import re
import sys

import pandas as pd

CIRS_UNIT = os.getenv("CIRS_UNIT", "field").lower()
CIRS_INCLUDE_QUESTION = os.getenv("CIRS_INCLUDE_QUESTION", "0") == "1"
CIRS_MIN_CHARS = int(os.getenv("CIRS_MIN_CHARS", "3"))

# source column -> output column
META_COLUMNS = {
    "RespondentID": "respondent_id",
    "MeasurementName": "measurement_name",
    "Clinical Background": "clinical_background",
    "Clinical Background Raw": "clinical_background_raw",
    "Average Experience Rating": "experience_rating",
    "Your Specialty": "specialty",
    "Locations of Use": "locations_of_use",
    "EMR": "emr",
    "PrimaryWork Location": "work_location",
}

# source column -> short field key (used in document IDs)
TEXT_COLUMNS = {
    "Fix 1": "fix1",
    "Fix 2": "fix2",
    "Fix 3": "fix3",
    "Most valuable improvements": "most_valuable",
    "CommentsGeneral": "comments_general",
    "Disagree IT Seeking to Improve EHR Comments": "disagree_it",
    "IT Department Improve EHR Comments": "it_improve",
    "Useful Training Comment": "training",
    "AI greatest potential not currently used": "ai_potential",
    "Hardware issues description": "hardware",
}

PLACEHOLDERS = {"", "-", "--", "---", ".", "/", "nan", "none", "n/a", "na", "k.a.",
                "keine", "kein", "nein", "no", "nil", "x", "?"}


def _key(name):
    return re.sub(r"\s+", "", str(name)).lower()


def _resolve_columns(df):
    """Map expected names to the actual column names in the file."""
    actual = {_key(c): c for c in df.columns}
    found_meta = {actual[_key(c)]: out for c, out in META_COLUMNS.items() if _key(c) in actual}
    found_text = {actual[_key(c)]: k for c, k in TEXT_COLUMNS.items() if _key(c) in actual}
    missing = [c for c in list(META_COLUMNS) + list(TEXT_COLUMNS) if _key(c) not in actual]
    if missing:
        print(f"Warning: columns not found: {missing}")
    if not found_text:
        raise ValueError("No free-text columns found. Check the column names.")
    return found_meta, found_text


def _clean(val):
    if pd.isna(val):
        return ""
    s = re.sub(r"\s+", " ", str(val)).strip()
    if s.lower() in PLACEHOLDERS or len(s) < CIRS_MIN_CHARS:
        return ""
    return s


def preprocess(df):
    meta_cols, text_cols = _resolve_columns(df)
    docs = []
    for idx, row in df.iterrows():
        meta = {out: ("" if pd.isna(row[src]) else str(row[src]).strip())
                for src, out in meta_cols.items()}
        rid = meta.get("respondent_id") or f"row{idx + 1}"
        rid = re.sub(r"\.0$", "", rid)  # Excel numbers: 1234.0 -> 1234
        meta["respondent_id"] = rid
        answers = [(src, key, _clean(row[src])) for src, key in text_cols.items()]
        answers = [a for a in answers if a[2]]
        if not answers:
            continue

        base = {
            **meta,
            # standard columns of the incident format, so all scripts work
            "priority": "", "status": "", "assignment_group": "",
            "department": meta.get("specialty", ""), "created_at": "",
        }
        if CIRS_UNIT == "respondent":
            text = "\n\n".join(f"{src}: {t}" for src, _, t in answers)
            docs.append({"incident_id": rid, "question": "all",
                         "consolidated_text": text, **base})
        elif CIRS_UNIT == "field":
            for src, key, t in answers:
                text = f"FRAGE: {src}\nANTWORT: {t}" if CIRS_INCLUDE_QUESTION else t
                docs.append({"incident_id": f"{rid}_{key}", "question": src,
                             "consolidated_text": text, **base})
        else:
            raise ValueError(f"Unknown CIRS_UNIT: {CIRS_UNIT}. Use field or respondent.")

    out = pd.DataFrame(docs)
    first = ["incident_id", "respondent_id", "question", "consolidated_text"]
    return out[first + [c for c in out.columns if c not in first]]


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "cirs.xlsx"
    outp = sys.argv[2] if len(sys.argv) > 2 else "cleaned_cirs.csv"
    if inp.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(inp)
    else:
        df = pd.read_csv(inp, sep=None, engine="python", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {inp}")

    out = preprocess(df)
    out.to_csv(outp, index=False, encoding="utf-8")
    out.to_json(outp.replace(".csv", ".json"), orient="records", force_ascii=False, indent=2)
    print(f"Unit: {CIRS_UNIT}. Wrote {len(out)} documents from "
          f"{out['respondent_id'].nunique() if len(out) else 0} respondents to "
          f"{outp} and {outp.replace('.csv', '.json')}")
    if len(out) and CIRS_UNIT == "field":
        print("\nDocuments per question:")
        print(out["question"].value_counts().to_string())


if __name__ == "__main__":
    main()
