"""
Shared outputs for all approaches.

write_report() writes into out_dir:
  assignments.csv     one row per document: cluster_id, cluster_name, noise flag,
                      metadata, UMAP coordinates, text preview
  clusters.csv        one row per cluster: name, size, share, top-3 examples,
                      hypothesis clusters inside (new_topic=True: none inside)
  summary.txt         totals, hypothesis evaluation, clusters by size, parameters
  umap_clusters.png   2D UMAP plot, one color per cluster
  hypothesis_report.csv     per hypothesis ID: where it landed
  hypothesis_summary.csv    per hypothesis cluster: together, best cluster, shared with, LLM name check
  hypothesis_metrics.json   pairwise precision / recall / F1, ARI, noise share of hypothesis IDs
  reference_agreement.json/.csv   agreement with existing category columns (ref_*),
                      e.g. the CIRS categories: ARI, AMI, NMI, homogeneity,
                      completeness, purity
  reference_crosstab_<col>.csv    clusters x categories counts
  reference_heatmap_<col>.png     share of each category per cluster

Cluster id -1 always means noise (HDBSCAN noise or "Sonstiges").
"""

import os
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.profile import P

# Columns never copied as metadata into assignments.csv
RESERVED = {"incident_id", "consolidated_text", "cluster_id", "cluster_name", "is_noise",
            "umap_x", "umap_y", "umap_sup_x", "umap_sup_y", "text_preview"}


def _meta_cols(df):
    """All non-empty metadata columns, whatever the input format."""
    cols = []
    for c in df.columns:
        if c in RESERVED:
            continue
        vals = df[c].dropna().astype(str).str.strip()
        if (vals != "").any():
            cols.append(c)
    return cols
# Set from the --umap-* flags by common/cli.py
UMAP_SETTINGS = dict(metric="cosine", n_neighbors=30, min_dist=0.1, pca_dims=0,
                     supervised=True, target_weight=0.5, random_state=42)


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
# HYPOTHESIS EVALUATION
# ----------------------------------------------------------------------

def _per_hypothesis(lookup, names, noise_name, hypotheses):
    rows, summary = [], []
    cluster_hits = {}  # cid -> Counter(hypothesis cluster)
    for cat, ids in hypotheses.items():
        landed = []
        for i in ids:
            if i not in lookup:
                status, cid = "NOT IN DATASET", None
            else:
                cid = lookup[i]
                status = noise_name if cid == -1 else f"C{cid}"
                landed.append(cid)
                if cid != -1:
                    cluster_hits.setdefault(cid, Counter())[cat] += 1
            rows.append({"hypothesis_cluster": cat, "id": i,
                         "cluster_id": "" if cid is None else cid,
                         "cluster_name": "" if cid is None else names.get(cid, noise_name),
                         "status": status})
        present = len(landed)
        non_noise = Counter(c for c in landed if c != -1)
        if present == 0:
            verdict, best, best_n = "no IDs in dataset", None, 0
        elif not non_noise:
            verdict, best, best_n = "not found (all noise)", None, 0
        else:
            best, best_n = non_noise.most_common(1)[0]
            if present == 1:
                verdict = "found (1 ID only)"
            elif best_n >= 2 and best_n / present >= 0.5:
                verdict = "found"
            else:
                verdict = "split"
        summary.append({
            "hypothesis_cluster": cat, "verdict": verdict,
            "together": f"{best_n}/{present}",
            "best_cluster_id": "" if best is None else best,
            "best_cluster_name": "" if best is None else names.get(best, ""),
            "ids_in_noise": sum(1 for c in landed if c == -1),
            "ids_not_in_dataset": len(ids) - present,
            "all_clusters_hit": ", ".join(f"C{c} ({n})" for c, n in non_noise.most_common()),
        })
    # separation: which other hypothesis clusters share the best cluster
    for r in summary:
        best = r["best_cluster_id"]
        others = [] if best == "" else [c for c in cluster_hits.get(best, {}) if c != r["hypothesis_cluster"]]
        r["shared_with"] = ", ".join(others)
    return pd.DataFrame(rows), pd.DataFrame(summary), cluster_hits


def _pairwise_metrics(lookup, hypotheses):
    """Compare the clustering with the hypothesis clusters, on their IDs only.

    An ID in several hypothesis clusters counts as "together" with all of them.
    Noise IDs count as singletons (never together with anything).
    """
    cats = {}
    for cat, ids in hypotheses.items():
        for i in ids:
            if i in lookup:
                cats.setdefault(i, []).append(cat)
    items = sorted(cats)
    pred = {i: (lookup[i] if lookup[i] != -1 else f"noise_{i}") for i in items}
    tp = fp = fn = 0
    for a, b in combinations(items, 2):
        g = bool(set(cats[a]) & set(cats[b]))
        p = pred[a] == pred[b]
        tp += g and p
        fp += p and not g
        fn += g and not p
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    ari = None
    if len(items) >= 2:
        from sklearn.metrics import adjusted_rand_score
        ari = float(adjusted_rand_score([cats[i][0] for i in items], [str(pred[i]) for i in items]))
    r = lambda x: None if x is None else round(float(x), 3)
    return {
        "hypothesis_ids_in_dataset": len(items),
        "hypothesis_ids_in_noise": sum(1 for i in items if lookup[i] == -1),
        "pairs_together_in_hypotheses": tp + fn,
        "pairs_together_in_clustering": tp + fp,
        "pairwise_precision": r(prec), "pairwise_recall": r(rec), "pairwise_f1": r(f1),
        "ari": r(ari),
    }


def _name_check(summary, cluster_info):
    """LLM judges if the best cluster's name fits the hypothesis cluster name."""
    from common import vllm
    jobs = [r for _, r in summary.iterrows() if r["best_cluster_id"] != ""]

    def ask(r):
        info = cluster_info.get(int(r["best_cluster_id"]), {})
        prompt = f"""Eine Fachperson hat einige Dokumente von Hand zu einem
Hypothesen-Cluster zusammengefasst und ihm einen Namen gegeben. Ein
Clustering-Verfahren hat einen Cluster gefunden, in dem die meisten dieser
Dokumente liegen. Beschreiben beide Cluster dasselbe Thema?

Hypothesen-Cluster: {r['hypothesis_cluster']}
Cluster-Name: {info.get('cluster_name', '')}
Cluster-Beschreibung: {info.get('description', '') or info.get('root_cause', '')}

Gib JSON zurück:
{{"match": "passt" | "teilweise" | "passt nicht", "reason": "1 Satz auf Deutsch"}}
"""
        o = vllm.chat_json(P["system"], prompt,
                           validate=lambda o: o.get("match") in ("passt", "teilweise", "passt nicht"))
        return o["match"], str(o.get("reason", ""))

    res = vllm.map_concurrent(ask, jobs, desc="name-check")
    out = {}
    for r, x in zip(jobs, res):
        out[r["hypothesis_cluster"]] = ("error", str(x)[:100]) if isinstance(x, Exception) else x
    return out


def evaluate_hypotheses(df, names, cluster_info, noise_name, hypotheses, name_check=True):
    lookup = dict(zip(df["incident_id"].astype(str), df["cluster_id"].astype(int)))
    rows, summary, cluster_hits = _per_hypothesis(lookup, names, noise_name, hypotheses)
    metrics = _pairwise_metrics(lookup, hypotheses)
    summary["name_match"], summary["name_reason"] = "", ""
    if name_check:
        checks = _name_check(summary, cluster_info)
        summary["name_match"] = summary["hypothesis_cluster"].map(lambda c: checks.get(c, ("", ""))[0])
        summary["name_reason"] = summary["hypothesis_cluster"].map(lambda c: checks.get(c, ("", ""))[1])
        counts = Counter(m for m, _ in checks.values())
        metrics["name_check"] = {k: counts.get(k, 0) for k in ("passt", "teilweise", "passt nicht")}
    cols = ["hypothesis_cluster", "verdict", "together", "best_cluster_id", "best_cluster_name",
            "shared_with", "name_match", "name_reason", "ids_in_noise", "ids_not_in_dataset",
            "all_clusters_hit"]
    return rows, summary[cols], metrics, cluster_hits


# ----------------------------------------------------------------------
# AGREEMENT WITH EXISTING CATEGORIES (ref_* columns)
# ----------------------------------------------------------------------

REF_EXPLANATION = (
    "ARI / AMI: agreement corrected for chance (0 = random, 1 = identical). "
    "homogeneity: each cluster holds one category. completeness: each category "
    "lies in one cluster. purity: share of documents in the main category of "
    "their cluster.")


def reference_columns(df):
    """Columns ref_* with at least 2 categories."""
    cols = []
    for c in df.columns:
        if c.startswith("ref_"):
            v = df[c].fillna("").astype(str).str.strip()
            if v[v != ""].nunique() >= 2:
                cols.append(c)
    return cols


def reference_agreement(df, names, out_dir):
    """Compare clusters (noise excluded) with every ref_* column."""
    from sklearn import metrics as skm
    results, tops = {}, {}
    for col in reference_columns(df):
        short = col[len("ref_"):]
        ref = df[col].fillna("").astype(str).str.strip()
        use = (df["cluster_id"] != -1) & (ref != "")
        y_true, y_pred = ref[use].to_numpy(), df.loc[use, "cluster_id"].to_numpy()
        if len(y_true) < 2:
            continue
        ct = pd.crosstab(pd.Series(y_pred, name="cluster_id"), pd.Series(y_true, name=short))
        h, c, v = skm.homogeneity_completeness_v_measure(y_true, y_pred)
        results[short] = {
            "categories": int(ct.shape[1]),
            "documents_compared": int(use.sum()),
            "documents_without_category": int((ref == "").sum()),
            "documents_in_noise": int(((df["cluster_id"] == -1) & (ref != "")).sum()),
            "ari": round(float(skm.adjusted_rand_score(y_true, y_pred)), 3),
            "ami": round(float(skm.adjusted_mutual_info_score(y_true, y_pred)), 3),
            "nmi": round(float(skm.normalized_mutual_info_score(y_true, y_pred)), 3),
            "homogeneity": round(float(h), 3),
            "completeness": round(float(c), 3),
            "v_measure": round(float(v), 3),
            "purity": round(float(ct.max(axis=1).sum() / ct.values.sum()), 3),
            "inverse_purity": round(float(ct.max(axis=0).sum() / ct.values.sum()), 3),
        }
        # counts, with cluster names
        out = ct.copy()
        out.insert(0, "cluster_name", [names.get(i, "") for i in out.index])
        out.to_csv(out_dir / f"reference_crosstab_{short}.csv", encoding="utf-8")
        # per category: in which cluster does it mainly land?
        cat_rows = []
        for cat in ct.columns:
            colv = ct[cat]
            best = colv.idxmax()
            cat_rows.append({"category": cat, "documents": int(colv.sum()),
                             "main_cluster_id": int(best), "main_cluster_name": names.get(best, ""),
                             "share_in_main_cluster": round(float(colv.max() / colv.sum()), 3),
                             "clusters_used": int((colv > 0).sum())})
        pd.DataFrame(cat_rows).sort_values("documents", ascending=False).to_csv(
            out_dir / f"reference_categories_{short}.csv", index=False, encoding="utf-8")
        # per cluster: main category and its share (for clusters.csv)
        tops[short] = {int(i): (ct.loc[i].idxmax(), float(ct.loc[i].max() / ct.loc[i].sum()))
                       for i in ct.index}
        plot_reference_heatmap(ct, names, short, results[short],
                               out_dir / f"reference_heatmap_{short}.png")
    if results:
        with open(out_dir / "reference_agreement.json", "w", encoding="utf-8") as f:
            json.dump({"explanation": REF_EXPLANATION, "columns": results}, f,
                      indent=2, ensure_ascii=False)
        pd.DataFrame(results).T.rename_axis("reference").to_csv(
            out_dir / "reference_agreement.csv", encoding="utf-8")
    return results, tops


def plot_reference_heatmap(ct, names, short, res, out_png, max_cats=30):
    """Rows: clusters (by size), columns: the largest categories; colour = share
    of the cluster's documents in that category."""
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    cats = ct.sum(axis=0).sort_values(ascending=False).index[:max_cats]
    share = ct[cats].div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(0.45 * len(cats) + 6, 0.32 * len(share) + 3))
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(c)[:40] for c in cats], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"C{i} {names.get(i, '')}"[:45] + f" ({int(ct.loc[i].sum())})"
                        for i in share.index], fontsize=7)
    more = f", largest {len(cats)} of {ct.shape[1]} categories" if ct.shape[1] > len(cats) else ""
    ax.set_title(f"Clusters vs. existing categories '{short}'{more}\n"
                 f"ARI {res['ari']}, AMI {res['ami']}, homogeneity {res['homogeneity']}, "
                 f"completeness {res['completeness']}, purity {res['purity']}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="share of the cluster")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------

def _palette(n):
    cmaps = [plt.get_cmap(c) for c in ("tab20", "tab20b", "tab20c")]
    base = [cm(i) for cm in cmaps for i in range(cm.N)]
    return [base[i % len(base)] for i in range(n)]


def umap_2d(vectors, labels=None):
    """2D UMAP of the embeddings. With labels: supervised UMAP, where the
    cluster labels pull documents of the same cluster together (-1 = noise
    counts as unlabeled)."""
    import umap
    s = UMAP_SETTINGS
    X = vectors
    if s["pca_dims"] and X.shape[1] > s["pca_dims"] and X.shape[0] > s["pca_dims"]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=s["pca_dims"], random_state=s["random_state"]).fit_transform(X)
    reducer = umap.UMAP(
        n_components=2, metric=s["metric"],
        n_neighbors=min(s["n_neighbors"], max(2, X.shape[0] - 1)),
        min_dist=s["min_dist"], random_state=s["random_state"],
        target_metric="categorical", target_weight=s["target_weight"])
    if labels is None:
        return reducer.fit_transform(X)
    return reducer.fit_transform(X, y=np.asarray(labels))


def _densest_point(pts, radius):
    """Label position: centre of the largest clump of the cluster, i.e. the
    point with the most cluster points within radius, averaged over those
    points. Keeps the label on the main clump, not in a gap between clumps."""
    if len(pts) <= 2:
        return pts[:, 0].mean(), pts[:, 1].mean()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=radius).fit(pts)
    neigh = nn.radius_neighbors(pts, return_distance=False)
    i = int(np.argmax([len(n) for n in neigh]))
    c = pts[neigh[i]]
    return c[:, 0].mean(), c[:, 1].mean()


def plot_umap(coords, labels, names, title, out_png, noise_name):
    fig, ax = plt.subplots(figsize=(20, 12))
    cids = sorted(c for c in set(labels.tolist()) if c != -1)
    colors = _palette(len(cids))
    radius = 0.04 * float(np.ptp(coords, axis=0).max() or 1.0)

    noise = labels == -1
    if noise.any():
        ax.scatter(coords[noise, 0], coords[noise, 1], c="lightgray", s=6, alpha=0.35,
                   label=f"{noise_name} ({int(noise.sum())})", zorder=1)
    for cid, col in zip(cids, colors):
        m = labels == cid
        ax.scatter(coords[m, 0], coords[m, 1], color=col, s=8, alpha=0.7,
                   label=f"C{cid} {names.get(cid, '')} ({int(m.sum())})", zorder=2)
        cx, cy = _densest_point(coords[m], radius)
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
                 noise_name="Noise", params=None, extra_cols=None,
                 hypotheses=None, name_check=True):
    """
    df            incidents (incident_id, consolidated_text, metadata)
    labels        cluster id per incident, -1 = noise
    cluster_info  {cid: {"cluster_name": str, "description": str, ...}}
    vectors       embeddings for the UMAP plot (None = no plot)
    noise_name    how noise is called in this method (e.g. "Sonstiges")
    params        dict of run parameters for summary.txt
    extra_cols    {column_name: list/array} added to assignments.csv
    hypotheses          {name: [ids]} from common.hypotheses.load_hypotheses, or None
    name_check    ask the LLM if cluster names match the hypothesis cluster names
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    meta_cols = _meta_cols(df)
    labels = np.asarray(labels).astype(int)
    df["cluster_id"] = labels
    names = {cid: str(info.get("cluster_name", f"Cluster {cid}"))
             for cid, info in cluster_info.items()}
    df["cluster_name"] = [names.get(c, noise_name) if c != -1 else noise_name for c in labels]
    df["is_noise"] = labels == -1
    for k, v in (extra_cols or {}).items():
        df[k] = list(v)

    # Hypothesis evaluation (only if hypothesis IDs occur in this dataset)
    ids = set(df["incident_id"].astype(str))
    has_hyp = bool(hypotheses) and any(i in ids for v in hypotheses.values() for i in v)
    metrics, hyp_summary, cluster_hits = None, None, {}
    if has_hyp:
        hyp_rows, hyp_summary, metrics, cluster_hits = evaluate_hypotheses(
            df, names, cluster_info, noise_name, hypotheses, name_check)
        hyp_rows.to_csv(out_dir / "hypothesis_report.csv", index=False, encoding="utf-8")
        hyp_summary.to_csv(out_dir / "hypothesis_summary.csv", index=False, encoding="utf-8")
        with open(out_dir / "hypothesis_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    elif hypotheses:
        print("[report] no hypothesis IDs in this dataset, evaluation skipped")

    # UMAP
    if vectors is not None:
        print("[report] UMAP 2D for plot...")
        coords = umap_2d(vectors)
        df["umap_x"], df["umap_y"] = coords[:, 0], coords[:, 1]
        n_noise = int((labels == -1).sum())
        title = (f"{method}: {len(names)} clusters, "
                 f"{len(df) - n_noise}/{len(df)} {P['label_en'].lower()} clustered, "
                 f"{n_noise} {noise_name}")
        plot_umap(coords, labels, names, title, out_dir / "umap_clusters.png", noise_name)
        if UMAP_SETTINGS["supervised"] and len(names) > 1:
            print("[report] supervised UMAP 2D for plot...")
            sup = umap_2d(vectors, labels)
            df["umap_sup_x"], df["umap_sup_y"] = sup[:, 0], sup[:, 1]
            w = UMAP_SETTINGS["target_weight"]
            plot_umap(sup, labels, names,
                      title + f"\nsupervised UMAP (target_weight {w}): layout partly "
                      "follows the cluster labels",
                      out_dir / "umap_clusters_supervised.png", noise_name)

    # Assignments, one row per incident
    cols = ["incident_id", "cluster_id", "cluster_name", "is_noise"] \
        + meta_cols \
        + [c for c in (extra_cols or {}) if c not in meta_cols] \
        + [c for c in ("umap_x", "umap_y", "umap_sup_x", "umap_sup_y") if c in df.columns]
    df["text_preview"] = df["consolidated_text"].str.slice(0, 300)
    df[cols + ["text_preview"]].to_csv(out_dir / "assignments.csv", index=False,
                                       encoding="utf-8")

    # Agreement with existing categories (ref_* columns, e.g. CIRS)
    ref_results, ref_tops = reference_agreement(df, names, out_dir)

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
            "hypothesis_clusters": ", ".join(f"{c} ({n})" for c, n in hits.most_common()),
            "new_topic": (not hits) if has_hyp else "",
            **{f"ref_{k}_main": (f"{t[cid][0]} ({100 * t[cid][1]:.0f}%)" if cid in t else "")
               for k, t in ref_tops.items()},
            **info,
        })
    clusters = pd.DataFrame(rows).sort_values("size", ascending=False)
    clusters.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8")

    # Summary
    n_noise = int((labels == -1).sum())
    what = P["label_en"]
    lines = [
        f"Method: {method}",
        f"{what} total:     {len(df)}",
        f"{what} clustered: {len(df) - n_noise} ({100 * (len(df) - n_noise) / len(df):.1f}%)",
        f"{what} {noise_name}: {n_noise} ({100 * n_noise / len(df):.1f}%)",
        f"Clusters:            {len(names)}",
    ]
    if has_hyp:
        m = metrics
        lines += [
            f"New topics (no hypothesis ID inside): {int((clusters['new_topic'] == True).sum())}",
            "",
            f"Hypothesis clusters ({m['hypothesis_ids_in_dataset']} IDs in dataset, "
            f"{m['hypothesis_ids_in_noise']} of them {noise_name}):",
            f"  pairwise precision {m['pairwise_precision']}, recall {m['pairwise_recall']}, "
            f"F1 {m['pairwise_f1']}, ARI {m['ari']}",
        ]
        if "name_check" in m:
            nc = m["name_check"]
            lines.append(f"  LLM name check: {nc['passt']} passt, {nc['teilweise']} teilweise, "
                         f"{nc['passt nicht']} passt nicht")
        lines.append("")
        for _, r in hyp_summary.iterrows():
            best = f" -> C{r['best_cluster_id']} {r['best_cluster_name']}" if r["best_cluster_name"] else ""
            extra = f" [{r['name_match']}]" if r["name_match"] else ""
            shared = f" (shared with: {r['shared_with']})" if r["shared_with"] else ""
            lines.append(f"  {r['hypothesis_cluster']:<28} {r['verdict']:<22} {r['together']:>5}"
                         f"{best}{extra}{shared}")
    else:
        lines += ["Hypothesis evaluation: skipped (no hypotheses file or no hypothesis IDs in this dataset)"]
    if ref_results:
        lines += ["", "Agreement with existing categories (noise excluded):"]
        for k, r in ref_results.items():
            lines.append(f"  {k:<24} {r['categories']:>4} categories: ARI {r['ari']}, "
                         f"AMI {r['ami']}, homogeneity {r['homogeneity']}, "
                         f"completeness {r['completeness']}, purity {r['purity']}")
        lines.append("  " + REF_EXPLANATION)
    lines += ["", "Clusters (by size):"]
    for _, r in clusters.iterrows():
        tag = "NEW " if r["new_topic"] is True else "    "
        lines.append(f"  {tag}C{r['cluster_id']:<4} {r['size']:>5}  {r['cluster_name']}")
    if params:
        lines += ["", "Parameters:"] + [f"  {k} = {v}" for k, v in params.items()]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"method": method, "documents": len(df), "noise": n_noise,
                   "clusters": len(names), "params": {k: str(v) for k, v in (params or {}).items()}},
                  f, indent=2, ensure_ascii=False)
    print("\n" + text)
    print(f"\nOutputs in: {out_dir}")
