#!/usr/bin/env python3
"""
Approach 1: ClusterLLM, granularity part
(after Zhang et al., "ClusterLLM: Large Language Models as a Guide for Text
Clustering", EMNLP 2023, https://github.com/zhang-yu-wei/ClusterLLM).

The LLM decides how fine the clusters are, instead of HDBSCAN parameters:
  1. Embed incidents (same API + cache as the other approaches),
     optional dim reduction (default UMAP to 10 dims).
  2. Build a full agglomerative (Ward) tree.
  3. For every merge step between K_MAX and K_MIN clusters, sample pairs
     (one incident from each of the two merged clusters) and ask the LLM:
     "same topic?"  -> stages/pairs.csv
  4. For every candidate k, count how many LLM answers agree with cutting the
     tree at k (merges above k should be "same", merges at or below k
     should be "different"). Pick the k with the highest agreement.
     -> k_scores.csv, k_scores.png
  5. Cut the tree at k. The LLM names each cluster from its most central
     incidents. Every incident gets a cluster (no noise).
  6. common/report.py: assignments, clusters, UMAP plot, validation.

Triplet fine-tuning of the embedder (ClusterLLM "perspective" part) is not
included.

Run from the folder that holds cleaned_incidents.json:
  python <repo>/clusterllm_granularity/run.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import vllm                                          # noqa: E402
from common.data import load_cleaned, truncate                   # noqa: E402
from common.embeddings import embed_texts, reduce_dimensions     # noqa: E402
from common.report import write_report, relabel_by_size          # noqa: E402

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "cluster_output/clusterllm_granularity"))
STAGE_DIR = OUTPUT_DIR / "stages"
RESUME = os.getenv("RESUME", "1") == "1"
SEED = int(os.getenv("SEED", "42"))

DIM_REDUCTION = os.getenv("DIM_REDUCTION", "umap")
DIMRED_N_COMPONENTS = int(os.getenv("DIMRED_N_COMPONENTS", "10"))
DIMRED_UMAP_N_NEIGHBORS = int(os.getenv("DIMRED_UMAP_N_NEIGHBORS", "15"))

K_MIN = int(os.getenv("K_MIN", "10"))
K_MAX = int(os.getenv("K_MAX", "100"))
PAIRS_PER_MERGE = int(os.getenv("PAIRS_PER_MERGE", "5"))
PAIR_MAX_CHARS = int(os.getenv("PAIR_MAX_CHARS", "1000"))
FORCE_K = int(os.getenv("FORCE_K", "0"))  # >0: skip the LLM pair stage

NAME_SAMPLES = int(os.getenv("NAME_SAMPLES", "8"))
NAME_MAX_CHARS = int(os.getenv("NAME_MAX_CHARS", "1000"))

GRANULARITY_QUESTION = os.getenv(
    "GRANULARITY_QUESTION",
    "Gehören diese beiden Meldungen zur selben Problemkategorie, d.h. "
    "betreffen sie dasselbe System bzw. dieselbe Funktion und dieselbe Art "
    "von Problem? Unterschiedliche Patienten, Personen oder Zeitpunkte spielen "
    "keine Rolle.",
)

DOMAIN = (
    "Du analysierst IT-Incident-Meldungen eines Spitals (Service Desk, "
    "klinische Systeme wie EPIC, Medikation, Geräte, Drucker usw.). "
    "Antworte ausschliesslich mit gültigem JSON, ohne Markdown."
)


# ----------------------------------------------------------------------
# DIM REDUCTION (cached: UMAP is not always repeatable across runs, and
# saved pair answers are only valid for the same tree)
# ----------------------------------------------------------------------

def stage_reduce(vectors):
    import hashlib
    key = hashlib.sha256(vectors.tobytes()).hexdigest()[:12] + \
        f"_{DIM_REDUCTION}_{DIMRED_N_COMPONENTS}_{DIMRED_UMAP_N_NEIGHBORS}_{SEED}"
    path = STAGE_DIR / f"reduced_{key}.npy"
    if RESUME and path.exists():
        print(f"[dimred] reuse {path}")
        return np.load(path)
    X = reduce_dimensions(vectors, DIM_REDUCTION, DIMRED_N_COMPONENTS,
                          DIMRED_UMAP_N_NEIGHBORS, 0.0, SEED)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X


# ----------------------------------------------------------------------
# TREE
# ----------------------------------------------------------------------

def build_tree(X):
    from scipy.cluster.hierarchy import linkage, to_tree
    print(f"[tree] Ward linkage on {X.shape}")
    Z = linkage(X, method="ward")
    _, nodes = to_tree(Z, rd=True)
    return Z, nodes


def merge_steps(Z, nodes, n):
    """Merges from j clusters to j-1, for j in K_MIN+1 .. K_MAX.

    Linkage row r creates node n+r; after row r there are n-r-1 clusters,
    so the merge j -> j-1 is row n-j.
    """
    steps = []
    for j in range(min(K_MAX, n), K_MIN, -1):
        r = n - j
        a, b = int(Z[r, 0]), int(Z[r, 1])
        steps.append({"j": j, "a": nodes[a].pre_order(), "b": nodes[b].pre_order()})
    return steps


# ----------------------------------------------------------------------
# PAIRS
# ----------------------------------------------------------------------

def _pair_prompt(t1, t2):
    return f"""{GRANULARITY_QUESTION}

Meldung A:
{truncate(t1, PAIR_MAX_CHARS)}

Meldung B:
{truncate(t2, PAIR_MAX_CHARS)}

Gib JSON zurück: {{"same": true}} oder {{"same": false}}
"""


def stage_pairs(df, steps):
    path = STAGE_DIR / "pairs.csv"
    if RESUME and path.exists():
        old = pd.read_csv(path)
        sides = {s["j"]: (set(s["a"]), set(s["b"])) for s in steps}
        ok = len(old) > 0 and all(
            r.j in sides and r.i1 < len(df) and r.i2 < len(df)
            and df["incident_id"].iat[r.i1] == r.incident_1
            and df["incident_id"].iat[r.i2] == r.incident_2
            and r.i1 in sides[r.j][0] and r.i2 in sides[r.j][1]
            for r in old.itertuples())
        if ok:
            print(f"[pairs] reuse {path}")
            return old
        print("[pairs] saved pairs do not match the current tree, asking the LLM again")

    rng = random.Random(SEED)
    pairs = []
    for s in steps:
        for _ in range(PAIRS_PER_MERGE):
            pairs.append({"j": s["j"], "i1": rng.choice(s["a"]), "i2": rng.choice(s["b"])})
    print(f"[pairs] {len(pairs)} LLM pair questions "
          f"({len(steps)} merges x {PAIRS_PER_MERGE})")

    texts = df["consolidated_text"].tolist()

    def run(p):
        obj = vllm.chat_json(DOMAIN, _pair_prompt(texts[p["i1"]], texts[p["i2"]]),
                             validate=lambda o: isinstance(o.get("same"), bool))
        return obj["same"]

    res = vllm.map_concurrent(run, pairs, desc="pairs")
    out = pd.DataFrame(pairs)
    out["incident_1"] = df["incident_id"].values[out["i1"]]
    out["incident_2"] = df["incident_id"].values[out["i2"]]
    out["same"] = [None if isinstance(r, Exception) else bool(r) for r in res]
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    return out


def choose_k(pairs):
    p = pairs.dropna(subset=["same"]).copy()
    p["same"] = p["same"].astype(str).str.lower().isin(["true", "1"])
    p_same = p.groupby("j")["same"].mean()  # share of "same" per merge j -> j-1
    js = sorted(p_same.index)
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        agree = sum(p_same[j] for j in js if j > k) + sum(1 - p_same[j] for j in js if j <= k)
        rows.append({"k": k, "agreement": agree / max(1, len(js))})
    scores = pd.DataFrame(rows)
    best = int(scores.loc[scores["agreement"].idxmax(), "k"])
    return best, scores, p_same


def plot_k(scores, p_same, best, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(scores["k"], scores["agreement"], color="tab:blue", label="Agreement with LLM")
    ax1.axvline(best, color="tab:red", linestyle="--", label=f"chosen k = {best}")
    ax1.set_xlabel("Number of clusters k")
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Agreement (share of merges)")
    ax2 = ax1.twinx()
    ax2.bar(p_same.index, p_same.values, color="tab:gray", alpha=0.3, width=0.8,
            label="LLM 'same' share of merge k->k-1")
    ax2.set_ylabel("LLM 'same' share")
    ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax1.set_title("ClusterLLM granularity: choice of k", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# NAMING
# ----------------------------------------------------------------------

def _name_prompt(texts):
    items = "\n---\n".join(truncate(t, NAME_MAX_CHARS) for t in texts)
    return f"""Diese Meldungen gehören zu einem Cluster. Benenne das gemeinsame Thema.

Gib JSON zurück:
{{"cluster_name": "2 bis 6 Wörter, Deutsch",
  "description": "1 Satz: welches Problem die Meldungen gemeinsam haben"}}

Meldungen:
{items}
"""


def name_clusters(df, labels, X):
    texts = df["consolidated_text"].tolist()
    cids = sorted(set(labels.tolist()))
    jobs = []
    for cid in cids:
        idx = np.where(labels == cid)[0]
        c = X[idx].mean(axis=0)
        order = idx[np.argsort(np.linalg.norm(X[idx] - c, axis=1))]
        jobs.append([texts[i] for i in order[:NAME_SAMPLES]])
    res = vllm.map_concurrent(
        lambda t: vllm.chat_json(DOMAIN, _name_prompt(t),
                                 validate=lambda o: bool(o.get("cluster_name"))),
        jobs, desc="naming")
    info = {}
    for cid, r in zip(cids, res):
        if isinstance(r, Exception):
            info[cid] = {"cluster_name": f"Cluster {cid} (unbenannt)", "description": ""}
        else:
            info[cid] = {"cluster_name": str(r["cluster_name"]).strip(),
                         "description": str(r.get("description", "")).strip()}
    return info


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    from scipy.cluster.hierarchy import cut_tree
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned()
    n = len(df)

    vectors = embed_texts(df["consolidated_text"].tolist())
    X = stage_reduce(vectors)
    Z, nodes = build_tree(X)

    if FORCE_K > 0:
        k = FORCE_K
        print(f"[k] FORCE_K = {k}, LLM pair stage skipped")
    else:
        steps = merge_steps(Z, nodes, n)
        pairs = stage_pairs(df, steps)
        k, scores, p_same = choose_k(pairs)
        scores.to_csv(OUTPUT_DIR / "k_scores.csv", index=False)
        plot_k(scores, p_same, k, OUTPUT_DIR / "k_scores.png")
        print(f"[k] chosen k = {k} (agreement {scores['agreement'].max():.3f})")

    labels = cut_tree(Z, n_clusters=k).ravel()
    labels, _ = relabel_by_size(labels)
    info = name_clusters(df, labels, X)

    write_report(
        df, labels, info, OUTPUT_DIR,
        method="ClusterLLM granularity", vectors=vectors, noise_name="Noise",
        params={
            "LLM": vllm.VLLM_LLM_MODEL, "Embedding": vllm.VLLM_EMBED_MODEL,
            "DIM_REDUCTION": f"{DIM_REDUCTION} ({DIMRED_N_COMPONENTS} dims)",
            "K range": f"{K_MIN}-{K_MAX}", "PAIRS_PER_MERGE": PAIRS_PER_MERGE,
            "chosen k": k, "FORCE_K": FORCE_K or "off", "SEED": SEED,
        },
    )


if __name__ == "__main__":
    main()
