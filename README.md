# LLM document clustering

This repo clusters hospital IT incident reports by topic. It compares
several approaches. All of them use a local vLLM server (an
OpenAI-compatible API) for embeddings and chat.

| Folder | Approach |
|---|---|
| `naive/` | embed, then HDBSCAN, then the LLM names each cluster (optional UMAP before HDBSCAN) |
| `llm_classification/` | the LLM generates topics, merges them, and classifies each incident ([details](llm_classification/README.md)) |
| `clusterllm_granularity/` | Ward tree; LLM pair questions choose the number of clusters ([details](clusterllm_granularity/README.md)) |
| `common/` | shared vLLM client, embedding cache, report and validation |
| `docs/approaches.md` | comparison of the approaches |
| `tests/` | mock vLLM server + synthetic data for a smoke test |

## Quick start

```bash
pip install -r requirements.txt
```

Preprocess the Excel export once. This writes `cleaned_incidents.json`:

```bash
python naive/preprocess_incidents.py
```

Set the same environment variables as in `naive/README.md`, section Setup.
Then run one or both approaches:

```bash
python llm_classification/run.py
```

```bash
python clusterllm_granularity/run.py
```

Both approaches run their LLM calls concurrently. Set the number of parallel
requests with `LLM_CONCURRENCY` (default 8).

## Input formats

The pipelines read one cleaned JSON file (`CLEANED_FILE`). Two preprocessors
write it:

| Format | Preprocessor | Output | `DATA_PROFILE` |
|---|---|---|---|
| IT incidents (Excel export) | `naive/preprocess_incidents.py` | `cleaned_incidents.json` | `incidents` (default) |
| CIRS / EHR survey (Excel or CSV) | `naive/preprocess_cirs.py` | `cleaned_cirs.json` | `cirs` |

`DATA_PROFILE` adapts the wording of the LLM prompts to the kind of text.

CIRS example:

```bash
python naive/preprocess_cirs.py cirs.xlsx cleaned_cirs.csv
```

```bash
export CLEANED_FILE=cleaned_cirs.json DATA_PROFILE=cirs
```

```bash
python llm_classification/run.py
```

The CIRS preprocessor:
- Clusters each non-empty free-text answer as one document
  (`CIRS_UNIT=field`, the default). The ID is `<RespondentID>_<field>`,
  for example `50000_fix1`. One respondent writes about several topics, so
  this gives cleaner topics.
- With `CIRS_UNIT=respondent`, it joins all answers of a person into one
  document instead.
- Drops empty answers and placeholders like `-` or `keine`.
- Keeps all metadata columns (specialty, clinical background, rating, ...).
  They appear in `assignments.csv`.
- With `CIRS_INCLUDE_QUESTION=1`, it prefixes each answer with its question.

The hypothesis validation runs only if the example INC IDs occur in the
data, so it is skipped for CIRS.

## Outputs

Each approach writes to `cluster_output/<approach>/`:

| File | Content |
|---|---|
| `assignments.csv` | one row per incident: `cluster_id`, `cluster_name`, `is_noise`, metadata, UMAP coordinates, text preview |
| `clusters.csv` | one row per cluster: name, size, share, top 3 example INCs, matched hypothesis categories, `new_topic` |
| `summary.txt` | incidents clustered vs. noise, clusters by size, hypothesis validation, parameters |
| `umap_clusters.png` | 2D UMAP plot with cluster names and sizes |
| `validation_report.csv` | per hypothesis example INC: the cluster it landed in |
| `validation_summary.csv` | per hypothesis category: verdict, best cluster, number of hits |

Cluster `-1` is noise: HDBSCAN noise, or "Sonstiges" in the LLM
classification.

## Hypothesis validation

The example INCs per category are in `common/hypotheses.py`. They are used
only after clustering, never as input. For each category:

| Verdict | Meaning |
|---|---|
| `found` | at least 2 examples, and at least half of them, are in one cluster |
| `split` | the examples are spread over several clusters |
| `not found (all noise)` | all examples in the dataset are noise |
| `no examples in dataset` | none of the example INCs is in the data |

A cluster that contains no example INC is marked `new_topic` in
`clusters.csv`. These are topics outside your hypotheses.

## Smoke test (no vLLM needed)

```bash
bash tests/smoke_test.sh
```

The test runs all three pipelines on synthetic data in both formats (400
incidents, 120 survey respondents) against a mock server. It checks only
that the code runs. It does not check the quality of the clusters.
