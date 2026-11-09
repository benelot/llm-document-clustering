"""
Embeddings with a disk cache shared by all approaches.

The cache key is a hash of model name + prefix + all texts, so a changed
input or model gives a new cache file. Progress is saved per chunk, so a
crashed run resumes.

  EMBED_CACHE_DIR   default: cluster_output/embed_cache
"""

import os
import json
import hashlib
from pathlib import Path

import numpy as np

from common import vllm

EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", "cluster_output/embed_cache"))


def _hash(texts):
    h = hashlib.sha256()
    h.update(vllm.VLLM_EMBED_MODEL.encode("utf-8"))
    h.update(b"\x01")
    h.update(vllm.EMBED_PREFIX.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_texts(texts):
    """Return L2-normalized embeddings (n, d) as float32."""
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _hash(texts)
    npy = EMBED_CACHE_DIR / f"embeddings_{key}.npy"
    meta = EMBED_CACHE_DIR / f"embeddings_{key}.meta.json"

    done = np.load(npy) if npy.exists() else None
    start = 0 if done is None else done.shape[0]
    if start >= len(texts):
        print(f"[embed] cache hit: {npy}")
        vectors = done[:len(texts)]
    else:
        if start:
            print(f"[embed] resuming at {start}/{len(texts)}")
        parts = [done] if done is not None else []
        bs = vllm.EMBED_API_BATCH_SIZE
        chunk = bs * max(1, vllm.LLM_CONCURRENCY)
        for c0 in range(start, len(texts), chunk):
            batches = [texts[i:i + bs] for i in range(c0, min(c0 + chunk, len(texts)), bs)]
            res = vllm.map_concurrent(vllm.embed_batch, batches, desc="embed")
            for r in res:
                if isinstance(r, Exception):
                    raise r
            arr = np.asarray([v for r in res for v in r], dtype=np.float32)
            parts.append(arr)
            np.save(npy, np.vstack(parts))
            print(f"[embed] {min(c0 + chunk, len(texts))}/{len(texts)}")
        vectors = np.vstack(parts)
        with open(meta, "w") as f:
            json.dump({"model": vllm.VLLM_EMBED_MODEL, "count": len(texts),
                       "shape": list(vectors.shape)}, f, indent=2)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def reduce_dimensions(X, method="none", n_components=10, n_neighbors=15,
                      min_dist=0.0, random_state=42):
    """Optional reduction before clustering: none | pca | umap."""
    method = (method or "none").lower()
    if method == "none":
        return X
    n_comp = min(n_components, X.shape[1], max(2, X.shape[0] - 2))
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_comp, random_state=random_state)
        out = pca.fit_transform(X)
        print(f"[dimred] PCA {X.shape[1]} -> {n_comp} "
              f"(explained variance {pca.explained_variance_ratio_.sum():.1%})")
        return out.astype(np.float32)
    if method == "umap":
        import umap
        out = umap.UMAP(n_components=n_comp, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="cosine", random_state=random_state,
                        n_jobs=1).fit_transform(X)
        print(f"[dimred] UMAP {X.shape[1]} -> {n_comp}")
        return out.astype(np.float32)
    raise ValueError(f"Unknown dim reduction: {method}. Use none, pca or umap.")
