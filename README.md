# LLM document clustering

This repo clusters hospital reports by topic. It compares three approaches
on two input formats: ServiceNow incidents and CIRS reports. All approaches
use a local vLLM server (an OpenAI-compatible API) for embeddings and chat.

| Folder | Approach |
|---|---|
| `naive/` | embed, then HDBSCAN, then the LLM names each cluster. Also holds the two preprocessors. |
| `llm_classification/` | the LLM generates topics, merges them, and classifies each document ([details](llm_classification/README.md)) |
| `clusterllm_granularity/` | Ward tree; LLM pair questions choose the number of clusters ([details](clusterllm_granularity/README.md)) |
| `common/` | shared flags, vLLM client, embedding cache, report and gold-standard evaluation |
| `gold/` | gold standard: hand-made groups of document IDs per category |
| `docs/approaches.md` | comparison of the approaches |
| `tests/` | mock vLLM server + synthetic data for a smoke test |

## Quick start

```bash
pip install -r requirements.txt
```

The API key comes only from the environment, so it does not end up in the
shell history:

```bash
export VLLM_API_KEY="your-secret"
```

Preprocess the export once. Pick the line for your format:

```bash
python naive/preprocess_incidents.py --input incidents.xlsx
```

```bash
python naive/preprocess_cirs.py --input cirs.xlsx
```

Then run any approach with the same flags:

```bash
python naive/cluster_incidents.py --format incidents --embed-url http://<host>:<port>/v1/embeddings --llm-url http://<host>:<port>/v1/chat/completions
```

```bash
python llm_classification/run.py --format cirs --embed-url ... --llm-url ...
```

```bash
python clusterllm_granularity/run.py --format incidents --embed-url ... --llm-url ...
```

Run `--help` on any script to see every flag with its default. Every flag
also falls back to an environment variable, which `--help` shows as
`[env: NAME]`. Environment variables from the old setup still work.

## Shared flags

| Flag | Default | Meaning |
|---|---|---|
| `--format` | `incidents` | `incidents` (ServiceNow) or `cirs`; also adapts the LLM prompt wording |
| `--input` | `cleaned_<format>.json` | cleaned input from the preprocessor |
| `--output-dir` | `cluster_output/<format>/<approach>` | output folder |
| `--gold-file` | `gold/<format>.json` | gold standard; the evaluation is skipped if the file is missing |
| `--embed-url`, `--embed-model` | localhost, `Qwen/Qwen3-Embedding-4B` | embedding endpoint |
| `--llm-url`, `--llm-model` | localhost, `Qwen/Qwen3.8-27B` | chat endpoint |
| `--llm-concurrency` | `8` | parallel requests; vLLM batches them on the server |
| `--llm-name-check` / `--no-llm-name-check` | on | LLM judges if cluster names match the gold categories |
| `--resume` / `--no-resume` | on | reuse saved stage results |
| `--plot` / `--no-plot` | on | UMAP plot |
| `--seed` | `42` | random seed |

## Input formats

| Format | Preprocessor | One document = |
|---|---|---|
| ServiceNow incidents (Excel) | `naive/preprocess_incidents.py` | one incident: short description + description + work notes |
| CIRS (Excel or CSV) | `naive/preprocess_cirs.py` | one CIRS report: all free-text answers, each prefixed with its question |

The CIRS preprocessor:
- Uses the RespondentID as the document ID.
- Drops empty answers and placeholders like `-` or `keine`.
- Keeps all metadata columns (specialty, clinical background, rating and
  more). They appear in `assignments.csv`.
- With `--unit field`, it makes one document per answer instead. The ID
  is then `<RespondentID>_<field>`, for example `50000_fix1`.

## Outputs

Each approach writes to `cluster_output/<format>/<approach>/`:

| File | Content |
|---|---|
| `assignments.csv` | one row per document: `cluster_id`, `cluster_name`, `is_noise`, metadata, UMAP coordinates, text preview |
| `clusters.csv` | one row per cluster: name, description, size, share, top 3 examples, gold categories inside, `new_topic` |
| `summary.txt` | documents clustered vs. noise, gold-standard evaluation, clusters by size, parameters |
| `umap_clusters.png` | 2D UMAP plot with cluster names and sizes |
| `gold_report.csv` | per gold ID: the cluster it landed in |
| `gold_summary.csv` | per gold category: verdict, together, best cluster, shared with, LLM name check |
| `gold_metrics.json` | pairwise precision, recall, F1, ARI, name check counts |

Cluster `-1` is noise: HDBSCAN noise, or "Sonstiges" in the LLM
classification.

The naive approach also writes its original files:
`cluster_summaries.csv` / `.json` and `cluster_assignments.csv`.

## Gold standard

A person read documents and grouped them into categories by hand. The file
is `gold/<format>.json`:

```json
{"Warnmeldung": ["INC1486211", "INC1486534"], "Pumpen / Perfusor": ["INC1514480"]}
```

The approaches never see this file. It is used only at the end, to measure
how close the clustering is to the hand-made groups.

**Per category** (`gold_summary.csv`):

| Column | Meaning |
|---|---|
| `together` | IDs in the most common cluster / IDs in the dataset |
| `verdict` | `found` (at least 2 IDs and at least half in one cluster), `split`, `not found (all noise)`, `no IDs in dataset` |
| `shared_with` | other gold categories in the same cluster; empty means cleanly separated |
| `name_match` | the LLM compares category name and cluster name: `passt`, `teilweise` or `passt nicht`, with a reason |

**Overall** (`gold_metrics.json`), computed on the gold IDs only:

| Metric | Meaning |
|---|---|
| pairwise precision | of the ID pairs the clustering puts together, the share that the gold standard also puts together |
| pairwise recall | of the ID pairs the gold standard puts together, the share that the clustering also puts together |
| pairwise F1 | precision and recall combined |
| ARI | agreement corrected for chance: 0 = random, 1 = identical |

- Noise counts as "not together with anything".
- An ID that is in several categories counts as together with all of them.

A cluster that contains no gold ID is marked `new_topic` in `clusters.csv`.
These are topics outside the hand-made categories.

`gold/cirs.json` does not exist yet. Until it does, the evaluation is
skipped for CIRS.

## Smoke test (no vLLM needed)

```bash
bash tests/smoke_test.sh
```

The test runs all three pipelines on synthetic data in both formats (400
incidents, 120 CIRS reports) against a mock server. It checks only that
the code runs. It does not check the quality of the clusters.
