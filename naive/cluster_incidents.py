#!/usr/bin/env python3
"""
Cluster hospital incident reports using:
- BGE embeddings via local API or local sentence-transformers
- HDBSCAN / DBSCAN clustering
- Qwen LLM summaries via vLLM

Expected input:
- cleaned_incidents.json

Outputs:
- cluster_output/cluster_summaries.csv
- cluster_output/cluster_summaries.json
- cluster_output/cluster_assignments.csv
"""

import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CLEANED_FILE = os.getenv("CLEANED_FILE", "cleaned_incidents.json")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "cluster_output"))

# Embedding backend: "api" or "local"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "api").lower()

# If EMBED_BACKEND=api
VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8000/v1/embeddings")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "BAAI/bge-m3")

# If EMBED_BACKEND=local
BGE_LOCAL_MODEL = os.getenv("BGE_LOCAL_MODEL", "BAAI/bge-m3")
BGE_LOCAL_BATCH_SIZE = int(os.getenv("BGE_LOCAL_BATCH_SIZE", "32"))
BGE_MAX_SEQ_LEN = int(os.getenv("BGE_MAX_SEQ_LEN", "1024"))

# Optional prefix for embeddings, e.g. "Represent this hospital incident report for clustering: "
EMBED_PREFIX = os.getenv("EMBED_PREFIX", "")

# Qwen via vLLM
VLLM_LLM_URL = os.getenv("VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions")
VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "qwen:3.8:27b")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# API settings
EMBED_API_BATCH_SIZE = int(os.getenv("EMBED_API_BATCH_SIZE", "16"))
EMBED_DELAY_SECONDS = float(os.getenv("EMBED_DELAY_SECONDS", "0.0"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))

# Clustering
CLUSTER_ALGORITHM = os.getenv("CLUSTER_ALGORITHM", "hdbscan")  # "hdbscan" or "dbscan"
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "10"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))
DBSCAN_EPS_PERCENTILE = float(os.getenv("DBSCAN_EPS_PERCENTILE", "75"))

# LLM labeling
MIN_LABEL_CLUSTER_SIZE = int(os.getenv("MIN_LABEL_CLUSTER_SIZE", "5"))
MAX_INCIDENTS_PER_CLUSTER = int(os.getenv("MAX_INCIDENTS_PER_CLUSTER", "8"))
MAX_CHARS_PER_INCIDENT = int(os.getenv("MAX_CHARS_PER_INCIDENT", "1500"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
JSON_MODE = os.getenv("JSON_MODE", "0").lower() == "1"


# ----------------------------------------------------------------------
# HTTP HELPER
# ----------------------------------------------------------------------

def post_json(url, payload, timeout=TIMEOUT_SECONDS):
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY and VLLM_API_KEY.lower() != "empty":
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)

    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from {url}\n"
            f"Response: {response.text[:1000]}"
        )

    return response.json()


# ----------------------------------------------------------------------
# EMBEDDING VIA API
# ----------------------------------------------------------------------

def extract_embeddings(payload):
    """
    Robustly extract embeddings from different possible API response formats.
    """
    if isinstance(payload, dict) and "data" in payload:
        items = payload["data"]
    elif isinstance(payload, list):
        items = payload
    else:
        items = [payload]

    # If items contain 'index', sort by index to preserve input order
    if items and all(isinstance(x, dict) and "index" in x for x in items):
        items = sorted(items, key=lambda x: x.get("index", 0))

    out = []

    for item in items:
        if isinstance(item, dict):
            emb = (
                item.get("embedding")
                or item.get("embeddings")
                or item.get("values")
                or []
            )
        else:
            emb = item

        arr = np.asarray(emb, dtype=np.float32).flatten()

        if arr.size == 0:
            raise ValueError("Empty embedding returned")

        out.append(arr)

    return out

def _cache_npy():
    return OUTPUT_DIR / "embeddings.npy"

def _cache_meta():
    return OUTPUT_DIR / "embeddings.meta.json"

def _compute_hash(texts):
    h = hashlib.sha256()
    h.update(EMBED_PREFIX.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def _save_cache(arr, texts):
    np.save(_cache_npy(), arr)
    meta = {
        "hash": _compute_hash(texts),
        "count": len(texts),
        "shape": list(arr.shape),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_cache_meta(), "w") as f:
        json.dump(meta, f, indent=2)

def _load_cache(texts):
    """Return cached array or None if missing/stale."""
    npy_p, meta_p = _cache_npy(), _cache_meta()
    if not npy_p.exists() or not meta_p.exists():
        return None
    with open(meta_p) as f:
        meta = json.load(f)
    if meta.get("hash") != _compute_hash(texts):
        print("[cache] Input changed. Re-embedding everything.")
        return None
    arr = np.load(npy_p)
    print(f"[cache] Loaded {arr.shape[0]}/{len(texts)} cached embeddings.")
    return arr

def embed_texts_api(texts):
    cached = _load_cache(texts)

    if cached is not None and cached.shape[0] >= len(texts):
        print(f"[embed-api] Full cache hit. Skipping API calls.")
        vectors = cached
    else:
        # Start from cache if partial, else from scratch
        if cached is not None:
            all_embs = [cached[i] for i in range(cached.shape[0])]
            start_idx = len(all_embs)
            print(f"[embed-api] Resuming from row {start_idx}.")
        else:
            all_embs = []
            start_idx = 0

        for start in range(start_idx, len(texts), EMBED_API_BATCH_SIZE):
            batch = texts[start:start + EMBED_API_BATCH_SIZE]
            if EMBED_PREFIX:
                batch = [f"{EMBED_PREFIX}{t}" for t in batch]

            payload = {
                "model": VLLM_EMBED_MODEL,
                "input": batch,
                "encoding_format": "float",
            }

            print(
                f"[embed-api] batch {start // EMBED_API_BATCH_SIZE + 1} "
                f"({start}-{start + len(batch)}/{len(texts)})"
            )

            response = post_json(VLLM_EMBED_URL, payload)
            all_embs.extend(extract_embeddings(response))

            if EMBED_DELAY_SECONDS > 0:
                time.sleep(EMBED_DELAY_SECONDS)

            # Save progress every batch so a crash resumes cleanly
            _save_cache(np.vstack(all_embs), texts)

        vectors = np.vstack(all_embs)

    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    return vectors

# ----------------------------------------------------------------------
# EMBEDDING VIA LOCAL GPU (SENTENCE-TRANSFORMERS)
# ----------------------------------------------------------------------

_local_embedder = None


def get_local_embedder():
    global _local_embedder

    if _local_embedder is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers or torch is not installed. "
                "Install them or set EMBED_BACKEND=api."
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embed-local] loading model from {BGE_LOCAL_MODEL} on {device}")

        _local_embedder = SentenceTransformer(
            BGE_LOCAL_MODEL,
            device=device,
            trust_remote_code=True
        )
        _local_embedder.max_seq_length = BGE_MAX_SEQ_LEN

    return _local_embedder


def embed_texts_local(texts):
    model = get_local_embedder()

    if EMBED_PREFIX:
        texts = [f"{EMBED_PREFIX}{t}" for t in texts]

    print(f"[embed-local] embedding {len(texts)} texts")

    vectors = model.encode(
        texts,
        batch_size=BGE_LOCAL_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    return vectors


def embed_texts(texts):
    if EMBED_BACKEND == "api":
        return embed_texts_api(texts)
    elif EMBED_BACKEND == "local":
        return embed_texts_local(texts)
    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}. Use 'api' or 'local'.")


# ----------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------

def dbscan_clustering(X, n):
    """
    DBSCAN fallback using adaptive epsilon from k-distance heuristic.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    if n < 10:
        return np.zeros(n, dtype=int)

    k = max(3, min(20, int(n ** 0.5)))
    if k >= n:
        k = max(1, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto").fit(X)
    distances, _ = nn.kneighbors(X)

    k_distances = distances[:, k]
    eps = float(np.percentile(k_distances, DBSCAN_EPS_PERCENTILE))

    if eps <= 1e-6:
        eps = 0.1

    min_samples = max(3, min(10, n // 20))

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return model.fit_predict(X)


def hdbscan_clustering(X, n):
    """
    Primary clustering using HDBSCAN.
    """
    try:
        import hdbscan
    except ImportError as e:
        raise RuntimeError("hdbscan is not installed. Install it or set CLUSTER_ALGORITHM=dbscan.") from e

    min_cluster = HDBSCAN_MIN_CLUSTER_SIZE if HDBSCAN_MIN_CLUSTER_SIZE > 0 else max(3, min(25, int(n * 0.02)))
    min_samples = HDBSCAN_MIN_SAMPLES if HDBSCAN_MIN_SAMPLES > 0 else max(2, min(5, n // 100))

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=min_samples,
        metric="euclidean",
#        cluster_selection_method="eom"
        cluster_selection_method="leaf"
    )

    labels = model.fit_predict(X)

    # If HDBSCAN produced almost all noise, fall back to DBSCAN
    if n > 50 and np.mean(labels != -1) < 0.05:
        print("[cluster] HDBSCAN produced mostly noise. Using DBSCAN fallback.")
        return dbscan_clustering(X, n)

    return labels.astype(int)


def cluster_vectors(X):
    n = X.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    if n < 3:
        return np.zeros(n, dtype=int)

    if CLUSTER_ALGORITHM.lower() == "hdbscan":
        try:
            return hdbscan_clustering(X, n)
        except Exception as e:
            print(f"[cluster] HDBSCAN failed: {e}. Falling back to DBSCAN.")

    return dbscan_clustering(X, n)


# ----------------------------------------------------------------------
# LLM HELPERS
# ----------------------------------------------------------------------
def llm_generate(messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    # Qwen3 thinking models spend tokens on a hidden reasoning chain and can
    # return content=null if that chain eats the budget. Disable thinking:
    # /no_think in the user turn is the portable, version-proof switch.
    for m in messages:
        if m["role"] == "user":
            m["content"] = "/no_think " + m["content"]

    payload = {
        "model": VLLM_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    data = post_json(VLLM_LLM_URL, payload, timeout=600)

    try:
        choice = data["choices"][0]
    except Exception as e:
        raise RuntimeError(f"Bad LLM response shape: {str(data)[:500]}") from e

    content = choice.get("message", {}).get("content")

    # Fail loudly instead of silently producing "Unlabeled cluster"
    if content is None or not content.strip():
        raise RuntimeError(
            "LLM returned empty content. finish_reason="
            f"{choice.get('finish_reason')}. The model is likely still in "
            "thinking mode or max_tokens is too low."
        )

    return content

def parse_json_response(text):
    if not text:
        return {}

    text = text.strip()

    # Remove markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_output": text}


def most_common(values):
    s = values.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    return s.mode().iloc[0]


# ----------------------------------------------------------------------
# CLUSTER SUMMARIZATION
# ----------------------------------------------------------------------

def fallback_summary(group, reason=""):
    return {
        "cluster_name": "Uncategorized cluster",
        "root_cause": "Not summarized" + (f": {reason}" if reason else ""),
        "affected_area": most_common(group["assignment_group"]),
        "recommended_measure": "Manual review required",
        "urgency": "medium",
        "rationale": "No LLM summary was generated.",
        "most_relevant_incident_ids": group["incident_id"].head(5).tolist()
    }


def label_cluster(group):
    if len(group) < MIN_LABEL_CLUSTER_SIZE:
        return fallback_summary(group, "Cluster too small for automatic summarization.")

    sample = group.head(MAX_INCIDENTS_PER_CLUSTER)

    incident_lines = []

    for _, row in sample.iterrows():
        text = str(row.get("consolidated_text", ""))
        if len(text) > MAX_CHARS_PER_INCIDENT:
            text = text[:MAX_CHARS_PER_INCIDENT] + "..."

        incident_lines.append(
            f"ID: {row.get('incident_id', '')}\n"
            f"Priority: {row.get('priority', '')}\n"
            f"Assignment group: {row.get('assignment_group', '')}\n"
            f"Department: {row.get('department', '')}\n"
            f"Status: {row.get('status', '')}\n"
            f"Text: {text}"
        )

    incident_block = "\n---\n".join(incident_lines)

    system_prompt = (
        "You are a hospital patient-safety analyst. "
        "You analyze incident reports and identify systemic risks that require hospital-wide action. "
        "Return ONLY valid JSON. Do not include markdown."
    )

    user_prompt = f"""
Analyze the following cluster of hospital incident reports.

Return JSON with exactly these keys:
- "cluster_name": short descriptive label
- "root_cause": likely common root cause or systemic failure
- "affected_area": department/system/area affected
- "recommended_measure": practical hospital-wide safety measure
- "urgency": one of "high", "medium", "low"
- "rationale": 1-2 sentences
- "most_relevant_incident_ids": up to 5 incident IDs most representative

Incidents:
{incident_block}
"""

    try:
        content = llm_generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result = parse_json_response(content)

        if "cluster_name" not in result or not result.get("cluster_name"):
            result["cluster_name"] = "Unlabeled cluster"

        result["raw_llm_output"] = content

    except Exception as e:
        result = fallback_summary(group, reason=str(e))

    result["cluster_size"] = len(group)
    result["sample_incident_ids"] = sample["incident_id"].tolist()
    result["incident_ids"] = group["incident_id"].head(100).tolist()

    return result


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def load_cleaned(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Run preprocessing first or set CLEANED_FILE environment variable."
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8")

    return pd.read_json(path, orient="records")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned incidents...")
    df = load_cleaned(CLEANED_FILE)

    if df.empty:
        print("No incidents loaded.")
        return

    df = df[
        df["consolidated_text"].fillna("").astype(str).str.strip() != ""
    ].reset_index(drop=True)

    print(f"Loaded {len(df)} non-empty incidents.")

    if df.empty:
        print("No usable text after filtering.")
        return

    texts = df["consolidated_text"].fillna("").astype(str).tolist()

    print(f"Generating embeddings using backend: {EMBED_BACKEND}")
    vectors = embed_texts(texts)

    print("Clustering embeddings...")
    labels = cluster_vectors(vectors)

    df["cluster_id"] = labels

    clustered_df = df[df["cluster_id"] != -1].copy()
    unclustered_count = int((df["cluster_id"] == -1).sum())

    print(f"Found {clustered_df['cluster_id'].nunique()} clusters.")
    print(f"Unclustered incidents: {unclustered_count}")

    summaries = []

    if not clustered_df.empty:
        for cluster_id, group in clustered_df.groupby("cluster_id"):
            print(
                f"Summarizing cluster {int(cluster_id)} "
                f"({len(group)} incidents) with Qwen..."
            )

            summary = label_cluster(group)
            summary["cluster_id"] = int(cluster_id)
            summaries.append(summary)

    # Flatten summary for CSV
    summary_flat = pd.DataFrame([
        {
            "cluster_id": s.get("cluster_id"),
            "cluster_name": s.get("cluster_name", ""),
            "root_cause": s.get("root_cause", ""),
            "affected_area": s.get("affected_area", ""),
            "recommended_measure": s.get("recommended_measure", ""),
            "urgency": s.get("urgency", ""),
            "rationale": s.get("rationale", ""),
            "cluster_size": s.get("cluster_size", 0)
        }
        for s in summaries
    ])

    # Sort by cluster size descending, then urgency high -> medium -> low
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    if not summary_flat.empty:
        summary_flat["_urgency_sort"] = summary_flat["urgency"].map(urgency_order).fillna(3)
        summary_flat = summary_flat.sort_values(
            by=["cluster_size", "_urgency_sort"],
            ascending=[False, True]
        ).drop(columns=["_urgency_sort"]).reset_index(drop=True)

    summary_csv_path = OUTPUT_DIR / "cluster_summaries.csv"
    summary_json_path = OUTPUT_DIR / "cluster_summaries.json"
    assignments_csv_path = OUTPUT_DIR / "cluster_assignments.csv"

    if not summary_flat.empty:
        summary_flat.to_csv(summary_csv_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=[
            "cluster_id",
            "cluster_name",
            "root_cause",
            "affected_area",
            "recommended_measure",
            "urgency",
            "rationale",
            "cluster_size"
        ]).to_csv(summary_csv_path, index=False, encoding="utf-8")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    df.to_csv(assignments_csv_path, index=False, encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  - {summary_csv_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {assignments_csv_path}")

    if not summary_flat.empty:
        print("\nTop clusters:")
        print(summary_flat.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
