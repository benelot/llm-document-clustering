#!/usr/bin/env python3
"""
Preprocess a hospital-wide CIRS export (Excel or CSV) into the shared cleaned
format, so that every pipeline in this repo can read it (--format cirs).

One document per CIRS case (ID = Fallnummer). The text joins the free-text
fields, each with its heading:
  Titel, Was ist passiert?, Was wurde getan?,
  Wie könnten ähnliche Ereignisse künftig verhindert werden?,
  Zusammenfassung / Wichtigstes

The existing categories are NOT part of the text. They are kept as reference
columns (ref_*) and only used afterwards, to measure how well the clusters
agree with them (common/report.py):
  ref_hauptkategorie            Hauptkategorie
  ref_subkategorie              Hauptkategorie / Subkategorie
  ref_subsubkategorie           Hauptkategorie / Subkategorie / Sub Subkategorie
  ref_massnahmenklassifikation  Massnahmenklassifikation

All other columns are kept as metadata (they appear in assignments.csv).

Examples:
  python preprocess_cirs.py --input cirs.xlsx
  python preprocess_cirs.py --input cirs.csv --fields title,event,summary

Output: cleaned_cirs.csv + cleaned_cirs.json. Then run any pipeline with
--format cirs.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli  # noqa: E402

# key -> (source column, heading in the document text)
TEXT_FIELDS = {
    "title": ("Falltitel", "Titel"),
    "event": ("Was ist passiert?", "Was ist passiert?"),
    "action": ("Was wurde getan?", "Was wurde getan?"),
    "prevention": ("Wie könnten ähnliche Ereignisse künftig verhindert werden?",
                   "Wie könnten ähnliche Ereignisse künftig verhindert werden?"),
    "summary": ("Zusammenfassung / Wichtigstes", "Zusammenfassung / Wichtigstes"),
}

# source column -> output column
META_COLUMNS = {
    "Fallnummer": "fallnummer",
    "Erfassungsdatum": "created_at",
    "Abschlussdatum": "abschlussdatum",
    "Status": "status",
    "cirs_verwertbar": "cirs_verwertbar",
    "Beruf": "beruf",
    "Häufigkeit": "haeufigkeit",
    "Relevanz": "relevanz",
    "spital": "spital",
    "medizinbereich": "medizinbereich",
    "Meldekreis": "meldekreis",
    "bereich": "bereich",
}

CATEGORY_COLUMNS = ["Hauptkategorie", "Subkategorie", "Sub Subkategorie",
                    "Massnahmenklassifikation"]

PLACEHOLDERS = {"", "-", "--", ".", "/", "nan", "none", "n/a", "na", "k.a.", "keine", "x", "?"}


def _key(name):
    return re.sub(r"\s+", "", str(name)).lower()


def _clean(val):
    if pd.isna(val):
        return ""
    s = re.sub(r"[ \t]+", " ", str(val)).strip()
    s = re.sub(r"\s*\n\s*", " ", s)
    return "" if s.lower() in PLACEHOLDERS else s


def _value(val):
    """Metadata value as clean string (dates without time 00:00:00, ids without .0)."""
    if pd.isna(val):
        return ""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d %H:%M") if (val.hour or val.minute) else val.strftime("%Y-%m-%d")
    s = str(val).strip()
    return re.sub(r"\.0$", "", s)


def preprocess(df, fields):
    actual = {_key(c): c for c in df.columns}
    col = lambda name: actual.get(_key(name))

    missing = [c for c in [v[0] for v in TEXT_FIELDS.values()] + list(META_COLUMNS)
               + CATEGORY_COLUMNS if col(c) is None]
    if missing:
        print(f"Warning: columns not found: {missing}")
    if col("Fallnummer") is None:
        raise ValueError("Column 'Fallnummer' not found.")

    docs, empty = [], 0
    for idx, row in df.iterrows():
        parts = []
        for key in fields:
            src, heading = TEXT_FIELDS[key]
            if col(src) is not None:
                t = _clean(row[col(src)])
                if t:
                    parts.append(f"{heading}: {t}")
        if not parts:
            empty += 1
            continue

        meta = {out: _value(row[col(src)]) if col(src) is not None else ""
                for src, out in META_COLUMNS.items()}
        cats = {c: (_clean(row[col(c)]) if col(c) is not None else "") for c in CATEGORY_COLUMNS}
        haupt, sub, subsub = cats["Hauptkategorie"], cats["Subkategorie"], cats["Sub Subkategorie"]
        docs.append({
            "incident_id": meta["fallnummer"] or f"row{idx + 1}",
            "consolidated_text": "\n\n".join(parts),
            **meta,
            # standard columns of the shared format
            "priority": "", "assignment_group": "", "department": meta["medizinbereich"],
            # reference categories: never part of the text, used only for evaluation
            "ref_hauptkategorie": haupt,
            "ref_subkategorie": " / ".join(x for x in (haupt, sub) if x) if sub else "",
            "ref_subsubkategorie": " / ".join(x for x in (haupt, sub, subsub) if x) if subsub else "",
            "ref_massnahmenklassifikation": cats["Massnahmenklassifikation"],
        })
    out = pd.DataFrame(docs)
    print(f"{len(out)} documents, {empty} rows without any text skipped")
    if out["incident_id"].duplicated().any():
        print(f"Warning: {int(out['incident_id'].duplicated().sum())} duplicate Fallnummern")
    return out


def main(argv=None):
    p = cli.make_parser(__doc__)
    cli.add_flag(p, "--input", "CIRS_INPUT", "cirs.xlsx", str, "CIRS export, .xlsx/.xls or .csv")
    cli.add_flag(p, "--output", "CIRS_OUTPUT", "cleaned_cirs.csv", str,
                 "output CSV; a .json with the same name is written too")
    cli.add_flag(p, "--fields", "CIRS_FIELDS", ",".join(TEXT_FIELDS), str,
                 f"text fields to join, comma-separated, from: {', '.join(TEXT_FIELDS)}")
    args = p.parse_args(argv)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    unknown = [f for f in fields if f not in TEXT_FIELDS]
    if unknown:
        raise SystemExit(f"error: unknown --fields {unknown}; use {list(TEXT_FIELDS)}")

    inp, outp = args.input, args.output
    if inp.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(inp)
    else:
        df = pd.read_csv(inp, sep=None, engine="python", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {inp}")

    out = preprocess(df, fields)
    out.to_csv(outp, index=False, encoding="utf-8")
    out.to_json(outp.replace(".csv", ".json"), orient="records", force_ascii=False, indent=2)
    print(f"Wrote {outp} and {outp.replace('.csv', '.json')}")
    for c in ("ref_hauptkategorie", "ref_subkategorie", "ref_subsubkategorie",
              "ref_massnahmenklassifikation"):
        n = out[c].replace("", pd.NA).dropna().nunique()
        print(f"  {c}: {n} categories, {int((out[c] != '').sum())} cases with a value")


if __name__ == "__main__":
    main()
