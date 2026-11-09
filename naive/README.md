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

All settings are command-line flags; run `python cluster_incidents.py --help`
for the full list. The old environment variables still work as fallback.
The API key comes only from the environment:

```bash
export VLLM_API_KEY="your-secret"
```

## Running the pipeline

Excel to cleaned JSON (first run or after a new export). Pick the line for
your format:

```bash
python preprocess_incidents.py --input incidents.xlsx
```

```bash
python preprocess_cirs.py --input cirs.xlsx
```

Embed, cluster, summarize and report:

```bash
python cluster_incidents.py --format incidents --embed-url http://<embed-host>:<embed-port>/v1/embeddings --llm-url http://<llm-host>:<llm-port>/v1/chat/completions
```

Use `--format cirs` for CIRS reports. It reads `cleaned_cirs.json` and adapts
the LLM prompt.

A full run on ~3,500 incidents:

| Phase | Time |
|---|---|
| Embedding (221 API batches) | ~5 min |
| Clustering | < 1 s |
| LLM summarization (one call per cluster) | 2–10 min |

## Output files

Default folder: `cluster_output/<format>/naive/`.

```
cluster_summaries.csv       # one row per cluster, sorted by size × urgency
cluster_summaries.json      # same + raw LLM response + incident IDs
cluster_assignments.csv     # every document with a cluster_id column
embeddings.npy              # raw vector matrix (~36 MB), cached
embeddings.meta.json        # hash + shape, for cache validation
+ shared report: assignments.csv, clusters.csv, summary.txt,
  umap_clusters.png, hypothesis_report.csv, hypothesis_summary.csv, hypothesis_metrics.json
```

The shared report is described in the root `README.md`, sections Outputs
and Hypothesis clusters.

- **`cluster_summaries.csv`**: for safety officers and management.
- **`cluster_summaries.json`**: for IT, audit trail, drill-down.
- **`cluster_assignments.csv`**: to filter the original data by cluster.

## Analysis scripts (older, kept as they are)

The shared report now covers these. They still read
`cluster_output/cluster_assignments_2.csv` (and `cluster_summaries_2.csv`),
so copy the pipeline output there first.

| Script | Purpose |
|---|---|
| `cluster_examples.py` | Top 3 INC IDs per cluster (highest priority first) → `cluster_examples_lookup.csv` |
| `plot_clusters.py` | PCA + UMAP 2D plot of the clusters → `umap_clusters.png` |

## Tuning

Defaults work for ~3,500 incidents.

| Flag | Default | Increase if… | Decrease if… |
|---|---|---|---|
| `--hdbscan-min-cluster-size` | `10` | too many tiny clusters | clusters too broad |
| `--hdbscan-min-samples` | `3` | clusters too fine-grained | — |
| `--hdbscan-selection-method` | `leaf` | — | — (`eom` gives fewer, larger clusters) |
| `--min-label-cluster-size` | `5` | LLM labels noise clusters | too few clusters summarized |
| `--max-incidents-per-cluster` | `8` | — | Qwen context too long |
| `--max-chars-per-incident` | `0` (none) | — | — (optional cap; texts are fitted to the context anyway) |
| `--llm-temperature` | `0.1` | — | summaries inconsistent |
| `--dbscan-eps-percentile` | `75` | — | fallback clusters too broad |
| `--dim-reduction` | `none` | — | — (`none`, `pca` or `umap`; reduces dims before clustering, `umap` recommended) |
| `--dimred-n-components` | `10` | clusters merge too much | too much noise |
| `--dimred-umap-n-neighbors` | `15` | clusters too fragmented | clusters too broad |
| `--dimred-umap-min-dist` | `0.0` | — | — (keep `0.0` for clustering) |

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
├── preprocess_cirs.py
├── cluster_incidents.py
├── cluster_examples.py
├── plot_clusters.py
├── incidents.xlsx / cirs.xlsx  # input (not in git)
├── cleaned_<format>.json       # generated (not in git)
└── cluster_output/             # generated (not in git)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `socket.gaierror: Temporary failure in name resolution` | Transient WSL2 DNS glitch. Re-run, or `sudo systemctl restart systemd-resolved` |
| `HTTP 401 Unauthorized` from vLLM | `VLLM_API_KEY` not set or wrong |
| `HDBSCAN produced mostly noise` | Script auto-falls back to DBSCAN. If still bad, lower `--hdbscan-min-cluster-size` to `5` |
| Qwen returns non-JSON | Script falls back to a "Manual review required" summary. Lower `--llm-temperature` to `0.0` |
| Embedding response has unexpected shape | Inspect the raw response of the embedding endpoint; `extract_embeddings()` handles the common formats |
