#!/usr/bin/env python3
"""
Compare all finished runs side by side (any approach, any perspective).

A run is a folder with assignments.csv and clusters.csv from common/report.py.
For each run it reads the cluster sizes and, if present, the evaluation
against the hypothesis clusters.

Output:
  <out-dir>/compare_runs.csv              one row per run
  <out-dir>/compare_runs.png              precision / recall / F1 / ARI per run
  <out-dir>/compare_hypotheses.csv        one row per hypothesis cluster and run
  <out-dir>/compare_hypotheses_wide.csv   hypothesis x run table, plus the best run
  <out-dir>/compare_hypotheses.png        hypothesis x run heatmap
  console tables

Examples:
  python compare_runs.py --format incidents
  python compare_runs.py --root cluster_output/incidents/llm_classification
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(d, root):
    a = pd.read_csv(d / "assignments.csv", usecols=["incident_id", "is_noise"])
    c = pd.read_csv(d / "clusters.csv")
    info = {}
    if (d / "run_info.json").exists():
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    params = info.get("params", {})
    n = len(a)
    row = {
        "run": str(d.relative_to(root)) if d != root else d.name,
        "method": info.get("method", ""),
        "perspective": params.get("perspective", ""),
        "documents": n,
        "clusters": len(c),
        "noise_pct": round(100 * a["is_noise"].astype(bool).mean(), 1),
        # share of the largest cluster: a high value points to a catch-all cluster
        "largest_cluster_pct": round(100 * c["size"].max() / n, 1) if len(c) else 0.0,
        "new_topics": int((c.get("new_topic") == True).sum()) if "new_topic" in c else "",
    }
    m = d / "hypothesis_metrics.json"
    if m.exists():
        met = json.loads(m.read_text(encoding="utf-8"))
        row.update({
            "precision": met.get("pairwise_precision"), "recall": met.get("pairwise_recall"),
            "f1": met.get("pairwise_f1"), "ari": met.get("ari"),
        })
        nc = met.get("name_check", {})
        row.update({"name_passt": nc.get("passt", ""), "name_teilweise": nc.get("teilweise", ""),
                    "name_passt_nicht": nc.get("passt nicht", "")})
    hs = d / "hypothesis_summary.csv"
    if hs.exists():
        v = pd.read_csv(hs)["verdict"].astype(str)
        row.update({"hyp_found": int(v.str.startswith("found").sum()),
                    "hyp_split": int((v == "split").sum()),
                    "hyp_not_found": int(v.str.startswith("not found").sum())})
    return row


NAME_SCORE = {"passt": 1.0, "teilweise": 0.5, "passt nicht": 0.0}
NAME_SYMBOL = {"passt": "+", "teilweise": "~", "passt nicht": "-"}


def hypothesis_table(runs, root, out):
    """Hypothesis x run: for each hypothesis cluster, how well each run matches it."""
    rows = []
    for d in runs:
        f = d / "hypothesis_summary.csv"
        if not f.exists():
            continue
        run = str(d.relative_to(root)) if d != root else d.name
        h = pd.read_csv(f, dtype=str).fillna("")
        for _, r in h.iterrows():
            n, _, tot = r["together"].partition("/")
            n, tot = int(n or 0), int(tot or 0)
            share = n / tot if tot else None
            nm = r.get("name_match", "")
            rows.append({
                "hypothesis_cluster": r["hypothesis_cluster"], "run": run,
                "together": r["together"], "together_share": share,
                "verdict": r["verdict"], "best_cluster": r["best_cluster_name"],
                "name_match": nm, "shared_with": r.get("shared_with", ""),
                # ranking: name fit first, then share together, minus a little per
                # other hypothesis in the same cluster
                "score": (None if share is None else round(
                    NAME_SCORE.get(nm, 0.0) + share
                    - 0.1 * len([x for x in r.get("shared_with", "").split(", ") if x]), 3)),
            })
    if not rows:
        print("No hypothesis evaluation found in the runs.")
        return
    long = pd.DataFrame(rows)
    long.to_csv(out / "compare_hypotheses.csv", index=False, encoding="utf-8")

    # wide view: cell = "3/4 +" (together, name check symbol)
    long["cell"] = (long["together"] + " " + long["name_match"].map(NAME_SYMBOL).fillna("")
                    + long["shared_with"].map(lambda x: "*" if x else ""))
    order = list(dict.fromkeys(long["hypothesis_cluster"]))
    run_order = list(dict.fromkeys(long["run"]))
    wide = long.pivot(index="hypothesis_cluster", columns="run", values="cell").reindex(order)[run_order]
    def best_runs(h):
        g = long[(long["hypothesis_cluster"] == h) & long["score"].notna()]
        if g.empty:
            return ""
        top = g[g["score"] == g["score"].max()]
        if len(top) == len(g) and len(g) > 1:
            return "all equal"
        return "; ".join(f"{r.run} ({r.best_cluster})" for r in top.itertuples())
    wide["best run"] = [best_runs(h) for h in wide.index]
    wide.to_csv(out / "compare_hypotheses_wide.csv", encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 60):
        print("\nHypothesis x run (together, name check: + passt, ~ teilweise, - passt nicht; "
              "* = cluster shared with other hypotheses)")
        print(wide.to_string())
    print(f"\nSaved: {out / 'compare_hypotheses.csv'}, {out / 'compare_hypotheses_wide.csv'}")

    # heatmap: colour = share together, text = together + name symbol
    share = long.pivot(index="hypothesis_cluster", columns="run",
                       values="together_share").reindex(order)[run_order].astype(float)
    cells = long.pivot(index="hypothesis_cluster", columns="run", values="cell").reindex(order)[run_order]
    fig, ax = plt.subplots(figsize=(1.6 * len(run_order) + 4, 0.55 * len(order) + 2.5))
    im = ax.imshow(share.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(run_order)):
            t = cells.iat[i, j]
            v = share.iat[i, j]
            ax.text(j, i, "" if pd.isna(t) else t, ha="center", va="center", fontsize=8,
                    color="white" if (not pd.isna(v) and v > 0.6) else "black")
    ax.set_xticks(range(len(run_order)))
    ax.set_xticklabels(run_order, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_title("Hypothesis clusters x runs\ncolour: share of hypothesis IDs together in one "
                 "cluster; text: together, name check (+ passt, ~ teilweise, - passt nicht), "
                 "* = cluster shared with other hypotheses",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share together")
    plt.tight_layout()
    plt.savefig(out / "compare_hypotheses.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'compare_hypotheses.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="incidents", help="incidents or survey")
    p.add_argument("--root", default=None,
                   help="folder to search for runs (default: cluster_output/<format>)")
    p.add_argument("--out-dir", default=None, help="where to write (default: --root)")
    args = p.parse_args()
    root = Path(args.root or f"cluster_output/{args.format}")
    out = Path(args.out_dir or root)
    runs = sorted({f.parent for f in root.rglob("assignments.csv") if (f.parent / "clusters.csv").exists()})
    if not runs:
        raise SystemExit(f"No runs found under {root}")

    df = pd.DataFrame([read_run(d, root) for d in runs])
    if "f1" in df:
        df = df.sort_values("f1", ascending=False, na_position="last")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "compare_runs.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))
    print(f"\nSaved: {out / 'compare_runs.csv'}")

    hypothesis_table(runs, root, out)

    metrics = [m for m in ("precision", "recall", "f1", "ari") if m in df and df[m].notna().any()]
    if metrics:
        d = df.dropna(subset=metrics, how="all").iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(d) + 2))
        h = 0.8 / len(metrics)
        colors = {"precision": "#4C78A8", "recall": "#F58518", "f1": "#54A24B", "ari": "#B279A2"}
        for i, m in enumerate(metrics):
            ys = [k + (i - (len(metrics) - 1) / 2) * h for k in range(len(d))]
            ax.barh(ys, d[m].fillna(0).astype(float), height=h, label=m, color=colors[m])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["run"], fontsize=8)
        ax.set_xlim(min(0, float(d[metrics].min().min())), 1)
        ax.set_xlabel("score (agreement with the hypothesis clusters)")
        ax.set_title("Runs compared against the hypothesis clusters", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "compare_runs.png", dpi=150)
        print(f"Saved: {out / 'compare_runs.png'}")


if __name__ == "__main__":
    main()
