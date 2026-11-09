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

Examples (run from the folder that holds the cleaned input):
  python <repo>/clusterllm_granularity/run.py --format incidents
  python <repo>/clusterllm_granularity/run.py --format cirs --k-min 5 --k-max 60
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import vllm                                          # noqa: E402
from common.data import load_cleaned                             # noqa: E402
from common.embeddings import embed_texts, reduce_dimensions     # noqa: E402
from common.report import write_report, relabel_by_size          # noqa: E402
from common.profile import P                                     # noqa: E402
from common.hypotheses import load_hypotheses                                # noqa: E402
from common import cli                                           # noqa: E402

# ----------------------------------------------------------------------
# CONFIG (set from command-line flags in main(); run with --help)
# ----------------------------------------------------------------------

OUTPUT_DIR = STAGE_DIR = None
RESUME, SEED, PLOT = True, 42, True
DIM_REDUCTION, DIMRED_N_COMPONENTS, DIMRED_UMAP_N_NEIGHBORS = "umap", 10, 15
K_MIN, K_MAX, PAIRS_PER_MERGE, PAIR_MAX_CHARS, FORCE_K = 10, 100, 5, 0, 0
NAME_SAMPLES, NAME_MAX_CHARS = 8, 0
GRANULARITY_QUESTION = P["granularity_question"]
DOMAIN = P["system"]


def parse_args(argv=None):
    p = cli.make_parser(__doc__)
    cli.add_common_args(p)
    g = p.add_argument_group("dimension reduction before the tree")
    cli.add_flag(g, "--dim-reduction", "DIM_REDUCTION", "umap", str, "none, pca or umap",
                 choices=["none", "pca", "umap"])
    cli.add_flag(g, "--dimred-n-components", "DIMRED_N_COMPONENTS", 10, int, "target dimensions")
    cli.add_flag(g, "--dimred-umap-n-neighbors", "DIMRED_UMAP_N_NEIGHBORS", 15, int,
                 "UMAP n_neighbors")
    g = p.add_argument_group("granularity (LLM pair questions)")
    cli.add_flag(g, "--k-min", "K_MIN", 10, int, "smallest candidate number of clusters")
    cli.add_flag(g, "--k-max", "K_MAX", 100, int, "largest candidate number of clusters")
    cli.add_flag(g, "--pairs-per-merge", "PAIRS_PER_MERGE", 5, int, "LLM questions per merge step")
    cli.add_flag(g, "--pair-max-chars", "PAIR_MAX_CHARS", 0, int,
                 "optional cap per document in a pair question, 0 = none")
    cli.add_flag(g, "--granularity-question", "GRANULARITY_QUESTION", None, str,
                 "the 'same category?' question; defines the granularity "
                 "(default: depends on --format, see common/profile.py)")
    cli.add_flag(g, "--force-k", "FORCE_K", 0, int, ">0 skips the pair stage and uses this k")
    g = p.add_argument_group("cluster naming")
    cli.add_flag(g, "--name-samples", "NAME_SAMPLES", 8, int, "central documents shown for naming")
    cli.add_flag(g, "--name-max-chars", "NAME_MAX_CHARS", 0, int,
                 "optional cap per document for naming, 0 = none")
    return cli.apply_common(p.parse_args(argv), "clusterllm_granularity")


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

{P['item']} A:
{t1}

{P['item']} B:
{t2}

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

    fixed = len(DOMAIN) + len(_pair_prompt("", "")) + 50
    cuts = []

    def run(p):
        (t1, t2), n_cut = vllm.fit_docs([texts[p["i1"]], texts[p["i2"]]], fixed,
                                        max_doc_chars=PAIR_MAX_CHARS)
        cuts.append(n_cut)
        obj = vllm.chat_json(DOMAIN, _pair_prompt(t1, t2),
                             validate=lambda o: isinstance(o.get("same"), bool))
        return obj["same"]

    res = vllm.map_concurrent(run, pairs, desc="pairs")
    print(f"[pairs] {sum(cuts)} documents cut to fit the context")
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
    items = "\n---\n".join(texts)
    return f"""Diese {P['items']} gehören zu einem Cluster. Benenne das gemeinsame Thema.

Gib JSON zurück:
{{"cluster_name": "2 bis 6 Wörter, Deutsch",
  "description": "1 Satz: welches Thema die {P['items']} gemeinsam haben"}}

{P['items']}:
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
    fixed = len(DOMAIN) + len(_name_prompt([])) + 50
    fitted = [vllm.fit_docs(j, fixed, per_item_chars=6, max_doc_chars=NAME_MAX_CHARS) for j in jobs]
    jobs = [t for t, _ in fitted]
    print(f"[naming] {sum(n for _, n in fitted)} documents cut to fit the context")
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

def main(argv=None):
    global OUTPUT_DIR, STAGE_DIR, RESUME, SEED, PLOT, DOMAIN, GRANULARITY_QUESTION
    global DIM_REDUCTION, DIMRED_N_COMPONENTS, DIMRED_UMAP_N_NEIGHBORS
    global K_MIN, K_MAX, PAIRS_PER_MERGE, PAIR_MAX_CHARS, FORCE_K, NAME_SAMPLES, NAME_MAX_CHARS
    from scipy.cluster.hierarchy import cut_tree
    args = parse_args(argv)
    cli.print_settings(args)
    cli.setup_llm_context(args)
    OUTPUT_DIR, STAGE_DIR = args.output_dir, args.output_dir / "stages"
    RESUME, SEED, PLOT, DOMAIN = args.resume, args.seed, args.plot, P["system"]
    GRANULARITY_QUESTION = args.granularity_question or P["granularity_question"]
    DIM_REDUCTION, DIMRED_N_COMPONENTS, DIMRED_UMAP_N_NEIGHBORS = \
        args.dim_reduction, args.dimred_n_components, args.dimred_umap_n_neighbors
    K_MIN, K_MAX, PAIRS_PER_MERGE, PAIR_MAX_CHARS, FORCE_K = \
        args.k_min, args.k_max, args.pairs_per_merge, args.pair_max_chars, args.force_k
    NAME_SAMPLES, NAME_MAX_CHARS = args.name_samples, args.name_max_chars

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned(args.input)
    n = len(df)

    vectors = embed_texts(df["consolidated_text"].tolist())
    X = stage_reduce(vectors)
    Z, nodes = build_tree(X)

    if FORCE_K > 0:
        k = FORCE_K
        print(f"[k] --force-k {k}, LLM pair stage skipped")
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
        method="ClusterLLM granularity", vectors=vectors if PLOT else None, noise_name="Noise",
        hypotheses=load_hypotheses(args.hypotheses_file), name_check=args.llm_name_check,
        params={
            "format": args.format, "input": args.input,
            "LLM": vllm.VLLM_LLM_MODEL, "Embedding": vllm.VLLM_EMBED_MODEL,
            "DIM_REDUCTION": f"{DIM_REDUCTION} ({DIMRED_N_COMPONENTS} dims)",
            "K range": f"{K_MIN}-{K_MAX}", "PAIRS_PER_MERGE": PAIRS_PER_MERGE,
            "chosen k": k, "force_k": FORCE_K or "off", "SEED": SEED,
        },
    )


if __name__ == "__main__":
    main()
