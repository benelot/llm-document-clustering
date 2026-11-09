# Hospital Incident Clustering (naive approach)

Groups hospital IT incident reports by semantic similarity and uses an LLM
to identify root causes and recommend hospital-wide safety measures.

## What it does

Reads incident reports from an Excel export of the IT service desk, groups
the ones that describe the **same underlying problem**, and for each group
asks a large language model to explain **what's going wrong** and
**what the hospital should do about it**.

The goal is to surface patterns no single person sees — e.g. 47 incidents
across 12 departments that all trace back to the same EPIC configuration
gap.

## How it works

### Step 1 — Preprocessing

Each Excel row is stitched into one coherent text: short description +
full description + work notes, with noise (timestamps, forwarding
chatter) stripped out. Metadata (priority, assignment group, department,
status) is preserved for filtering.

### Step 2 — Embedding

Each incident's text is converted to a 2,560-dim vector by
**Qwen3-Embedding-4B** (served via vLLM). Two incidents describing the
same problem land near each other in vector space, even if the wording
is completely different.

### Step 3 — Clustering

**HDBSCAN** finds dense regions in the vector space and calls them
clusters. It does not require you to specify the number of clusters.
Incidents that don't clearly belong to any group are labeled noise and
excluded. If HDBSCAN finds too few clusters it falls back to DBSCAN.

### Step 4 — LLM Summarization

For each cluster with ≥ 5 incidents, up to 8 representative incidents
are sent to **Qwen3.8-27B** (served via vLLM), which returns structured
JSON:

| Field | Example |
|---|---|
| `cluster_name` | "Perfusor unit mismatch: mg vs mcg for Remifentanyl" |
| `root_cause` | "EPIC Ultiva default is mg instead of mcg for Remifentanyl" |
| `affected_area` | "APPL_EPIC_WILLOW / Anästhesie" |
| `recommended_measure` | "Standardize opioid perfusor units to mcg in EPIC" |
| `urgency` | "high" |
| `rationale` | "12 of 18 incidents involved anesthesia sedation drugs" |

## Prerequisites

See `requirements.txt` (pandas, numpy, requests, openpyxl, scikit-learn,
hdbscan, umap-learn, matplotlib).

```bash
pip install -r requirements.txt
```

No GPU access needed from the client. All inference runs remotely on the
vLLM servers.

## Setup

```bash
# Embedding endpoint
export EMBED_BACKEND="api"
export VLLM_EMBED_URL="http://<embed-host>:<embed-port>/v1/embeddings"
export VLLM_EMBED_MODEL="Qwen/Qwen3-Embedding-4B"

# Chat endpoint
export VLLM_LLM_URL="http://<llm-host>:<llm-port>/v1/chat/completions"
export VLLM_LLM_MODEL="Qwen/Qwen3.8-27B"

# Auth
export VLLM_API_KEY="your-secret"

# Paths
export CLEANED_FILE="cleaned_incidents.json"
export OUTPUT_DIR="cluster_output"
```

## Running the pipeline

```bash
# 1. Excel → cleaned JSON  (first run or after new export)
python preprocess_incidents.py

# 2. Embed + cluster + summarize
python cluster_incidents.py
```

A full run on ~3,500 incidents:

| Phase | Time |
|---|---|
| Embedding (221 API batches) | ~5 min |
| Clustering | < 1 s |
| LLM summarization (one call per cluster) | 2–10 min |

## Analysis scripts

Run after the pipeline. They read `cluster_output/cluster_assignments_2.csv`
(and `cluster_summaries_2.csv`), so rename or copy the pipeline output first.

| Script | Purpose |
|---|---|
| `cluster_examples.py` | Top 3 INC IDs per cluster (highest priority first) → `cluster_examples_lookup.csv` |
| `hypothesis_matches.py` | Checks if hand-picked example INCs per hypothesis category land in the mapped clusters → `validation_report.csv` |
| `plot_clusters.py` | PCA + UMAP 2D plot of the clusters → `umap_clusters.png` |

```bash
python cluster_examples.py
python hypothesis_matches.py
python plot_clusters.py
```

## Output files

```
cluster_output/
├── cluster_summaries.csv       # one row per cluster, sorted by size × urgency
├── cluster_summaries.json      # same + raw LLM response + incident IDs
├── cluster_assignments.csv     # every incident with a cluster_id column
├── embeddings.npy              # raw vector matrix (~36 MB), cached
└── embeddings.meta.json        # hash + shape, for cache validation
```

- **`cluster_summaries.csv`** — for safety officers and management.
- **`cluster_summaries.json`** — for IT, audit trail, drill-down.
- **`cluster_assignments.csv`** — to filter the original data by cluster.

## Tuning

All via environment variables. Defaults work for ~3,500 incidents.

| Variable | Default | Increase if… | Decrease if… |
|---|---|---|---|
| `HDBSCAN_MIN_CLUSTER_SIZE` | `10` | too many tiny clusters | clusters too broad |
| `HDBSCAN_MIN_SAMPLES` | `3` | clusters too fine-grained | — |
| `MIN_LABEL_CLUSTER_SIZE` | `5` | LLM labels noise clusters | too few clusters summarized |
| `MAX_INCIDENTS_PER_CLUSTER` | `8` | — | Qwen context too long |
| `MAX_CHARS_PER_INCIDENT` | `1500` | — | important details cut off |
| `LLM_TEMPERATURE` | `0.1` | — | summaries inconsistent |
| `DBSCAN_EPS_PERCENTILE` | `75` | — | fallback clusters too broad |
| `DIM_REDUCTION` | `none` | — | — (`none`, `pca` or `umap`; reduces dims before clustering, `umap` recommended) |
| `DIMRED_N_COMPONENTS` | `10` | clusters merge too much | too much noise |
| `DIMRED_UMAP_N_NEIGHBORS` | `15` | clusters too fragmented | clusters too broad |
| `DIMRED_UMAP_MIN_DIST` | `0.0` | — | — (keep `0.0` for clustering) |

## Limitations

- **One-off incidents** are marked as noise and excluded. A single
  occurrence doesn't warrant a hospital-wide measure.
- **Root cause suggestions** are LLM-generated from the incident text.
  Treat as a triage aid, not a confirmed root cause analysis.
- **Volume changes.** HDBSCAN params are tuned for ~3,500 incidents.
  Re-tune for significantly larger or smaller datasets.
- **Language.** Optimized for German. Mixed-language text slightly
  reduces embedding coherence.

## Project structure

```
naive/
├── README.md
├── requirements.txt
├── preprocess_incidents.py
├── cluster_incidents.py
├── cluster_examples.py
├── hypothesis_matches.py
├── plot_clusters.py
├── incidents.xlsx              # input (not in git)
├── cleaned_incidents.json      # generated (not in git)
└── cluster_output/             # generated (not in git)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `socket.gaierror: Temporary failure in name resolution` | Transient WSL2 DNS glitch. Re-run, or `sudo systemctl restart systemd-resolved` |
| `HTTP 401 Unauthorized` from vLLM | `VLLM_API_KEY` not set or wrong |
| `HDBSCAN produced mostly noise` | Script auto-falls back to DBSCAN. If still bad, lower `HDBSCAN_MIN_CLUSTER_SIZE` to `5` |
| Qwen returns non-JSON | Script falls back to a "Manual review required" summary. Lower `LLM_TEMPERATURE` to `0.0` |
| Embedding response has unexpected shape | Inspect the raw response of the embedding endpoint; `extract_embeddings()` handles the common formats |
