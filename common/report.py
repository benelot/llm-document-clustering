"""
Shared outputs for all approaches.

write_report() writes into out_dir:
  assignments.csv          one row per incident: cluster_id, cluster_name, noise flag,
                           metadata, UMAP coordinates, text preview
  clusters.csv             one row per cluster: name, size, share, top-3 example INCs,
                           matched hypothesis categories ("" = new topic)
  summary.txt              totals: clustered vs. noise, number of clusters, parameters
  umap_clusters.png        2D UMAP plot, one color per cluster
  validation_report.csv    per hypothesis example INC: where it landed
  validation_summary.csv   per hypothesis category: found / split / not found

Cluster id -1 always means noise (HDBSCAN noise or "Sonstiges").
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.hypotheses import EXAMPLES

META_COLS = ["priority", "status", "assignment_group", "department", "created_at"]
UMAP_RANDOM_STATE = int(os.getenv("UMAP_RANDOM_STATE", "42"))


def relabel_by_size(labels):
    """Renumber clusters 0..k-1 by size (largest = 0). -1 stays -1."""
    labels = np.asarray(labels)
    counts = Counter(int(l) for l in labels if l != -1)
    order = [cid for cid, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    mapping = {old: new for new, old in enumerate(order)}
    mapping[-1] = -1
    return np.array([mapping[int(l)] for l in labels]), mapping


def _prio_rank(v):
    s = str(v)
    for i in range(1, 5):
        if str(i) in s:
            return i
    return 9


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------

def validate_hypotheses(df, names, noise_name):
    lookup = dict(zip(df["incident_id"].astype(str), df["cluster_id"].astype(int)))
    rows, summary = [], []
    cluster_hits = {}  # cid -> Counter(category)

    for cat, incs in EXAMPLES.items():
        landed = []
        for inc in incs:
            if inc not in lookup:
                status, cid = "NOT IN DATASET", None
            else:
                cid = lookup[inc]
                status = noise_name if cid == -1 else f"C{cid}"
                landed.append(cid)
                if cid != -1:
                    cluster_hits.setdefault(cid, Counter())[cat] += 1
            rows.append({
                "hypothesis_category": cat,
                "incident_id": inc,
                "cluster_id": "" if cid is None else cid,
                "cluster_name": "" if cid is None else names.get(cid, noise_name),
                "status": status,
            })

        present = len(landed)
        non_noise = Counter(c for c in landed if c != -1)
        if present == 0:
            verdict, best, best_n = "no examples in dataset", None, 0
        elif not non_noise:
            verdict, best, best_n = "not found (all noise)", None, 0
        else:
            best, best_n = non_noise.most_common(1)[0]
            if present == 1:
                verdict = "found (1 example only)"
            elif best_n >= 2 and best_n / present >= 0.5:
                verdict = "found"
            else:
                verdict = "split"
        summary.append({
            "hypothesis_category": cat,
            "verdict": verdict,
            "best_cluster_id": "" if best is None else best,
            "best_cluster_name": "" if best is None else names.get(best, ""),
            "examples_in_best_cluster": best_n,
            "examples_in_dataset": present,
            "examples_total": len(incs),
            "examples_in_noise": sum(1 for c in landed if c == -1),
            "all_clusters_hit": ", ".join(f"C{c} ({n})" for c, n in non_noise.most_common()),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary), cluster_hits


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------

def _palette(n):
    cmaps = [plt.get_cmap(c) for c in ("tab20", "tab20b", "tab20c")]
    base = [cm(i) for cm in cmaps for i in range(cm.N)]
    return [base[i % len(base)] for i in range(n)]


def umap_2d(vectors):
    from sklearn.decomposition import PCA
    import umap
    X = vectors
    if X.shape[1] > 50 and X.shape[0] > 50:
        X = PCA(n_components=50, random_state=UMAP_RANDOM_STATE).fit_transform(X)
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=UMAP_RANDOM_STATE).fit_transform(X)


def plot_umap(coords, labels, names, title, out_png, noise_name):
    fig, ax = plt.subplots(figsize=(20, 12))
    cids = sorted(c for c in set(labels.tolist()) if c != -1)
    colors = _palette(len(cids))

    noise = labels == -1
    if noise.any():
        ax.scatter(coords[noise, 0], coords[noise, 1], c="lightgray", s=6, alpha=0.35,
                   label=f"{noise_name} ({int(noise.sum())})", zorder=1)
    for cid, col in zip(cids, colors):
        m = labels == cid
        ax.scatter(coords[m, 0], coords[m, 1], color=col, s=8, alpha=0.7,
                   label=f"C{cid} {names.get(cid, '')} ({int(m.sum())})", zorder=2)
        cx, cy = np.median(coords[m, 0]), np.median(coords[m, 1])
        ax.annotate(f"C{cid}", xy=(cx, cy), fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8,
                              edgecolor=col, linewidth=1))

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_facecolor("#fafafa")
    ncol = 1 if len(cids) <= 35 else 2
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              framealpha=0.9, ncol=ncol, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------

def write_report(df, labels, cluster_info, out_dir, method, vectors=None,
                 noise_name="Noise", params=None, extra_cols=None):
    """
    df            incidents (incident_id, consolidated_text, metadata)
    labels        cluster id per incident, -1 = noise
    cluster_info  {cid: {"cluster_name": str, "description": str, ...}}
    vectors       embeddings for the UMAP plot (None = no plot)
    noise_name    how noise is called in this method (e.g. "Sonstiges")
    params        dict of run parameters for summary.txt
    extra_cols    {column_name: list/array} added to assignments.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    labels = np.asarray(labels).astype(int)
    df["cluster_id"] = labels
    names = {cid: str(info.get("cluster_name", f"Cluster {cid}"))
             for cid, info in cluster_info.items()}
    df["cluster_name"] = [names.get(c, noise_name) if c != -1 else noise_name for c in labels]
    df["is_noise"] = labels == -1
    for k, v in (extra_cols or {}).items():
        df[k] = list(v)

    # Validation
    val_rows, val_summary, cluster_hits = validate_hypotheses(df, names, noise_name)
    val_rows.to_csv(out_dir / "validation_report.csv", index=False, encoding="utf-8")
    val_summary.to_csv(out_dir / "validation_summary.csv", index=False, encoding="utf-8")

    # UMAP
    if vectors is not None:
        print("[report] UMAP 2D for plot...")
        coords = umap_2d(vectors)
        df["umap_x"], df["umap_y"] = coords[:, 0], coords[:, 1]
        n_noise = int((labels == -1).sum())
        title = (f"{method}: {len(names)} clusters, "
                 f"{len(df) - n_noise}/{len(df)} incidents clustered, "
                 f"{n_noise} {noise_name}")
        plot_umap(coords, labels, names, title, out_dir / "umap_clusters.png", noise_name)

    # Assignments, one row per incident
    cols = ["incident_id", "cluster_id", "cluster_name", "is_noise"] \
        + [c for c in META_COLS if c in df.columns] \
        + list((extra_cols or {}).keys()) \
        + [c for c in ("umap_x", "umap_y") if c in df.columns]
    df["text_preview"] = df["consolidated_text"].str.slice(0, 300)
    df[cols + ["text_preview"]].to_csv(out_dir / "assignments.csv", index=False,
                                       encoding="utf-8")

    # Cluster table
    rows = []
    for cid in sorted(names):
        g = df[df["cluster_id"] == cid]
        if "priority" in g.columns:
            g = g.assign(_pr=g["priority"].map(_prio_rank)).sort_values("_pr")
        hits = cluster_hits.get(cid, Counter())
        info = {k: v for k, v in cluster_info[cid].items() if k != "cluster_name"}
        rows.append({
            "cluster_id": cid,
            "cluster_name": names[cid],
            "size": len(g),
            "share_pct": round(100 * len(g) / len(df), 1),
            "top_examples": ", ".join(g["incident_id"].head(3)),
            "hypothesis_categories": ", ".join(f"{c} ({n})" for c, n in hits.most_common()),
            "new_topic": not hits,
            **info,
        })
    clusters = pd.DataFrame(rows).sort_values("size", ascending=False)
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8")

    # Summary
    n_noise = int((labels == -1).sum())
    lines = [
        f"Method: {method}",
        f"Incidents total:     {len(df)}",
        f"Incidents clustered: {len(df) - n_noise} ({100 * (len(df) - n_noise) / len(df):.1f}%)",
        f"Incidents {noise_name}: {n_noise} ({100 * n_noise / len(df):.1f}%)",
        f"Clusters:            {len(names)}",
        f"New topics (no hypothesis example inside): {int(clusters['new_topic'].sum())}",
        "",
        "Hypothesis validation:",
    ]
    for _, r in val_summary.iterrows():
        best = f" -> C{r['best_cluster_id']} {r['best_cluster_name']}" if r["best_cluster_name"] else ""
        lines.append(f"  {r['hypothesis_category']:<28} {r['verdict']:<24} "
                     f"{r['examples_in_best_cluster']}/{r['examples_in_dataset']}{best}")
    lines += ["", "Clusters (by size):"]
    for _, r in clusters.iterrows():
        tag = "NEW " if r["new_topic"] else "    "
        lines.append(f"  {tag}C{r['cluster_id']:<4} {r['size']:>5}  {r['cluster_name']}")
    if params:
        lines += ["", "Parameters:"] + [f"  {k} = {v}" for k, v in params.items()]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nOutputs in: {out_dir}")
