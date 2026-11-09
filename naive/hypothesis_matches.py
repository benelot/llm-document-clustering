#!/usr/bin/env python3
"""
Validates: do the human-provided example INC IDs for each hypothesis
category actually land in the algorithmic clusters we mapped to that
category?

Reads: cluster_output/cluster_assignments.csv
Outputs: console table + cluster_output/validation_report.csv
"""

import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

OUTPUT_DIR = Path("cluster_output")
ASSIGNMENTS_CSV = OUTPUT_DIR / "cluster_assignments_2.csv"

# Mapping: hypothesis category → set of cluster_ids we said match it
CLUSTER_MAP = {
    "Medikationsstammdaten":      {21, 20, 17, 18, 16, 15, 8, 13},
    "My Insel":                   set(),
    "Medikamente abgleichen":     set(),
    "Drucken und Scannen":        {26, 24, 5, 7, 23, 22, 19},
    "Ansichten und Darstellung":  {11, 16},
    "Hochrisikomedikamente":      {6, 14, 12, 24, 25},
    "Pumpen / Perfusor":          {4, 12, 14, 3},
    "Warnmeldung":                {9, 15, 8},
    "Bestellung / Verrechnung":   {2, 13},
}

# Human-provided example INC IDs per category
EXAMPLES = {
    "Medikationsstammdaten": [
        "INC1452522", "INC1472605", "INC1473389", "INC1459578",
        "INC1474943", "INC1486026", "INC1644685", "INC1491370",
    ],
    "My Insel": [
        "INC1611691",
    ],
    "Medikamente abgleichen": [
        "INC1642389", "INC1642786", "INC1463758",
    ],
    "Drucken und Scannen": [
        "INC165223", "INC1654755", "INC1655109", "INC1473802",
    ],
    "Ansichten und Darstellung": [
        "INC1504196", "INC1601487", "INC1519507", "INC1489233",
    ],
    "Hochrisikomedikamente": [
        "INC1492786", "INC1507214", "INC1519507", "INC1522159",
    ],
    "Pumpen / Perfusor": [
        "INC1514480", "INC1514530", "INC1517757",
    ],
    "Warnmeldung": [
        "INC1486211", "INC1486534", "INC1486700",
    ],
    "Bestellung / Verrechnung": [
        "INC1494864", "INC1523414", "INC1575272", "INC1662025", "INC1454464",
    ],
}

# Which cluster belongs to which category (for "wrong cluster" messages)
CLUSTER_TO_CATEGORY = {}
for cat, cids in CLUSTER_MAP.items():
    for cid in cids:
        CLUSTER_TO_CATEGORY[cid] = cat


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    df = pd.read_csv(ASSIGNMENTS_CSV, encoding="utf-8")
    # Build lookup: incident_id → cluster_id
    lookup = dict(zip(df["incident_id"].astype(str), df["cluster_id"]))

    results = []
    print("=" * 80)
    print("VALIDATION: Do human-picked examples land in the mapped clusters?")
    print("=" * 80)

    for category, incs in EXAMPLES.items():
        mapped = CLUSTER_MAP[category]
        print(f"\n{'─' * 70}")
        print(f"  {category}")
        print(f"  Mapped clusters: {sorted(mapped) if mapped else '(none)'}")
        print(f"{'─' * 70}")

        hits = 0
        noise = 0
        wrong = 0
        missing = 0

        for inc in incs:
            if inc not in lookup:
                status = "NOT IN DATASET"
                missing += 1
            else:
                cid = int(lookup[inc])
                if cid == -1:
                    status = "NOISE (unclustered)"
                    noise += 1
                elif cid in mapped:
                    status = f"HIT → C{cid}"
                    hits += 1
                else:
                    other_cat = CLUSTER_TO_CATEGORY.get(cid, "?")
                    status = f"WRONG CLUSTER → C{cid} ({other_cat})"
                    wrong += 1

            print(f"    {inc:<14} {status}")
            results.append({
                "hypothesis_category": category,
                "incident_id": inc,
                "cluster_id": lookup.get(inc, "N/A"),
                "status": status,
            })

        total = len(incs)
        print(f"\n    Summary: {hits}/{total} hits | {noise} noise | "
              f"{wrong} wrong-cluster | {missing} not-in-dataset")

    # Save
    out_csv = OUTPUT_DIR / "validation_report.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
