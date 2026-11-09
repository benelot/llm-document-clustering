#!/usr/bin/env python3
"""
UMAP visualization of incident clusters.

Reads:
  cluster_output/embeddings.npy
  cluster_output/cluster_assignments.csv
  cluster_output/cluster_summaries.csv

Output:
  cluster_output/umap_clusters.png

Run:
  python umap_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / WSL-friendly
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "cluster_output"))
EMBEDDINGS_NPY = OUTPUT_DIR / "embeddings.npy"
ASSIGNMENTS_CSV = OUTPUT_DIR / "cluster_assignments_2.csv"
SUMMARIES_CSV = OUTPUT_DIR / "cluster_summaries_2.csv"
OUTPUT_PNG = OUTPUT_DIR / "umap_clusters.png"

PCA_DIMS = 50
UMAP_RANDOM_STATE = 42
FIG_SIZE = (16, 12)
DOT_SIZE = 8
DPI = 150

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("Loading embeddings...")
    vectors = np.load(EMBEDDINGS_NPY)
    print(f"  shape: {vectors.shape}")

    print("Loading assignments...")
    df = pd.read_csv(ASSIGNMENTS_CSV, encoding="utf-8")
    labels = df["cluster_id"].values

    print("Loading cluster summaries...")
    summaries = pd.read_csv(SUMMARIES_CSV, encoding="utf-8")
    name_map = dict(zip(summaries["cluster_id"].astype(int),
                        summaries["cluster_name"].astype(str)))

    # PCA pre-reduction (speeds up UMAP, standard practice)
    from sklearn.decomposition import PCA
    print(f"PCA → {PCA_DIMS} dims...")
    pca = PCA(n_components=PCA_DIMS, random_state=UMAP_RANDOM_STATE)
    reduced = pca.fit_transform(vectors)
    print(f"  explained variance: {sum(pca.explained_variance_ratio_):.1%}")

    # UMAP
    import umap
    print("UMAP → 2D...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=UMAP_RANDOM_STATE,
        n_neighbors=15,
        min_dist=0.1,
    )
    coords = reducer.fit_transform(reduced)

    # ------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    unique_clusters = sorted(set(labels), key=lambda x: (x == -1, x))

    for cid in unique_clusters:
        mask = labels == cid
        n = mask.sum()
        pts = coords[mask]

        if cid == -1:
            label = f"Noise ({n})"
            color = "lightgray"
            alpha = 0.3
            zorder = 1
        else:
            label = name_map.get(cid, f"Cluster {cid}")
            color = None  # let matplotlib cycle
            alpha = 0.6
            zorder = 2

        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, s=DOT_SIZE, alpha=alpha,
                   label=label, zorder=zorder)

        # Annotate centroid for clusters with ≥ 5 points
        if n >= 5 and cid != -1:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.annotate(
                label,
                xy=(cx, cy),
                fontsize=9,
                fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.75,
                          edgecolor="gray", linewidth=0.5),
                zorder=5,
            )

    ax.set_title("Hospital Incident Clusters (UMAP)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_facecolor("#fafafa")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
