#!/usr/bin/env python3
"""Füllt die 'Beispiele'-Spalte: top-3 INC-IDs je Cluster aus cluster_assignments.csv."""
import pandas as pd
from pathlib import Path

df = pd.read_csv("cluster_output/cluster_assignments_2.csv", encoding="utf-8")

# Priorität anordnen: "1 - Hoch" < "2 - Mittel" < "3" < "4 - Niedrig"
def prio_rank(v):
    s = str(v)
    for i in range(1, 5):
        if str(i) in s:
            return i
    return 9

rows = []
for cid, g in df[df["cluster_id"] != -1].groupby("cluster_id"):
    g = g.copy()
    if "priority" in g.columns:
        g["_pr"] = g["priority"].map(prio_rank)
        g = g.sort_values("_pr")
    top = g["incident_id"].head(3).tolist()
    rows.append({
        "cluster_id": int(cid),
        "cluster_size": len(g),
        "Beispiele": ", ".join(top),
    })

out = pd.DataFrame(rows).sort_values("cluster_size", ascending=False)
print(out.to_string(index=False))
out.to_csv("cluster_output/cluster_examples_lookup.csv", index=False, encoding="utf-8")
print("\nGespeichert: cluster_output/cluster_examples_lookup.csv")
