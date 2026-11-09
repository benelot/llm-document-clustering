# LLM document clustering

This repo clusters hospital texts by topic. It compares three approaches on
two input formats: ServiceNow incidents and an EHR survey among clinicians.
All approaches use a local vLLM server (an OpenAI-compatible API) for
embeddings and chat. A CIRS format will follow.

| Folder | Approach |
|---|---|
| `naive/` | embed, then HDBSCAN, then the LLM names each cluster. Also holds the two preprocessors. |
| `llm_classification/` | the LLM generates topics, merges them, and classifies each document ([details](llm_classification/README.md)) |
| `clusterllm_granularity/` | Ward tree; LLM pair questions choose the number of clusters ([details](clusterllm_granularity/README.md)) |
| `common/` | shared flags, vLLM client, embedding cache, report and hypothesis evaluation |
| `hypotheses/` | hypothesis clusters: hand-made groups of document IDs, each with a name |
| `docs/approaches.md` | comparison of the approaches |
| `compare_runs.py` | compares all runs (approaches, perspectives) in one table and chart |
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
python naive/preprocess_survey.py --input survey.xlsx
```

Then run any approach with the same flags:

```bash
python naive/cluster_incidents.py --format incidents --embed-url http://<host>:<port>/v1/embeddings --llm-url http://<host>:<port>/v1/chat/completions
```

```bash
python llm_classification/run.py --format survey --embed-url ... --llm-url ...
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
| `--format` | `incidents` | `incidents` (ServiceNow) or `survey`; also adapts the LLM prompt wording |
| `--input` | `cleaned_<format>.json` | cleaned input from the preprocessor |
| `--output-dir` | `cluster_output/<format>/<approach>` | output folder |
| `--hypotheses-file` | `hypotheses/<format>.json` | hypothesis clusters; the evaluation is skipped if the file is missing |
| `--embed-url`, `--embed-model` | localhost, `Qwen/Qwen3-Embedding-4B` | embedding endpoint |
| `--llm-url`, `--llm-model` | localhost, `Qwen/Qwen3.8-27B` | chat endpoint |
| `--llm-concurrency` | `8` | parallel requests; vLLM batches them on the server |
| `--llm-thinking` / `--no-llm-thinking` | off | let the model think before answering; off sends `enable_thinking=false` |
| `--llm-context-tokens` | `0` = ask the server | context length of the chat model; read from `/v1/models` (`max_model_len`), fallback 32768 |
| `--llm-name-check` / `--no-llm-name-check` | on | LLM judges if cluster names match the hypothesis cluster names |
| `--resume` / `--no-resume` | on | reuse saved stage results |
| `--plot` / `--no-plot` | on | UMAP plot |
| `--seed` | `42` | random seed |
| `--embed-instruction` | `auto` | task instruction for Qwen3-Embedding (`Instruct: <task>\nQuery: <text>`), per format; `none` = off, or your own text |
| `--umap-n-neighbors` | `30` | UMAP plot: larger = more global structure |
| `--umap-metric` | `cosine` | UMAP plot: distance on the embeddings |
| `--umap-supervised` / `--no-umap-supervised` | on | also write the supervised plot |
| `--umap-target-weight` | `0.5` | supervised plot: 0 = labels ignored, 1 = labels dominate |

## Long documents

No document is cut at a fixed length. At startup, each script asks the vLLM
server for the context length of the chat model (`max_model_len` from
`/v1/models`). From this it computes how many characters fit into one
prompt. The answer budget (`--llm-max-tokens`) is kept free, and about 3
characters count as one token.

- Batches are packed up to this limit. If documents are long, a batch holds
  fewer documents than its maximum, for example `--classify-batch-size`.
- A document is only cut if it does not fit into one prompt on its own.
  The log reports how many were cut.
- The embeddings always get the full text.

If the server does not report the context length, set it with
`--llm-context-tokens`.

## Input formats

| Format | Preprocessor | One document = |
|---|---|---|
| ServiceNow incidents (Excel) | `naive/preprocess_incidents.py` | one incident: short description + description + work notes |
| EHR survey among clinicians (Excel or CSV) | `naive/preprocess_survey.py` | one respondent: all free-text answers, each prefixed with its question |

The survey preprocessor:
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
| `clusters.csv` | one row per cluster: name, description, size, share, top 3 examples, hypothesis clusters inside, `new_topic` |
| `summary.txt` | documents clustered vs. noise, hypothesis evaluation, clusters by size, parameters |
| `umap_clusters.png` | 2D UMAP of the embeddings, colored by cluster: shows how well the clusters agree with the embeddings |
| `umap_clusters_supervised.png` | same, but UMAP also uses the cluster labels: shows the clusters clearly and which ones are close; the separation is partly enforced, so it does not test the clustering |
| `hypothesis_report.csv` | per hypothesis ID: the cluster it landed in |
| `hypothesis_summary.csv` | per hypothesis cluster: verdict, together, best cluster, shared with, LLM name check |
| `hypothesis_metrics.json` | pairwise precision, recall, F1, ARI, name check counts |

Cluster `-1` is noise: HDBSCAN noise, or "Sonstiges" in the LLM
classification.

The naive approach also writes its original files:
`cluster_summaries.csv` / `.json` and `cluster_assignments.csv`.

## Comparing runs

```bash
python compare_runs.py --format incidents
```

The script finds every finished run under `cluster_output/<format>/`: all
approaches and all perspectives. It writes `compare_runs.csv` and
`compare_runs.png`, a bar chart of precision, recall, F1 and ARI per run.

| Column | Meaning |
|---|---|
| `clusters`, `noise_pct` | number of clusters, share of noise |
| `largest_cluster_pct` | share of the largest cluster; a high value points to a catch-all cluster |
| `new_topics` | clusters without any hypothesis ID |
| `precision`, `recall`, `f1`, `ari` | agreement with the hypothesis clusters |
| `hyp_found`, `hyp_split`, `hyp_not_found` | verdicts per hypothesis cluster |
| `name_passt`, `name_teilweise`, `name_passt_nicht` | LLM name check |

## Hypothesis clusters

A person read some documents and grouped a handful of them by hand into
named hypothesis clusters. The file is `hypotheses/<format>.json`:

```json
{"Warnmeldung": ["INC1486211", "INC1486534"], "Pumpen / Perfusor": ["INC1514480"]}
```

The approaches never see this file. They put every document into exactly
one cluster (or noise) on their own. At the end, the report tests two
things automatically:
- Do the IDs of one hypothesis cluster land in the same algorithmic cluster,
  and apart from the IDs of the other hypothesis clusters?
- Does the name of that algorithmic cluster fit the name of the hypothesis
  cluster? The LLM judges this.

The matching algorithmic cluster is the one that holds most of the
hypothesis IDs, so no map from hypothesis to cluster ID is needed.

**Per hypothesis cluster** (`hypothesis_summary.csv`):

| Column | Meaning |
|---|---|
| `together` | IDs in the most common algorithmic cluster / IDs in the dataset |
| `verdict` | `found` (at least 2 IDs and at least half in one cluster), `split`, `not found (all noise)`, `no IDs in dataset` |
| `shared_with` | other hypothesis clusters in the same algorithmic cluster; empty means cleanly separated |
| `name_match` | the LLM compares the two names: `passt`, `teilweise` or `passt nicht`, with a reason |

**Overall** (`hypothesis_metrics.json`), computed on the hypothesis IDs only:

| Metric | Meaning |
|---|---|
| pairwise precision | of the ID pairs the clustering puts together, the share that are also together in a hypothesis cluster |
| pairwise recall | of the ID pairs that are together in a hypothesis cluster, the share that the clustering also puts together |
| pairwise F1 | precision and recall combined |
| ARI | agreement corrected for chance: 0 = random, 1 = identical |

- Noise counts as "not together with anything".
- An ID that is in several hypothesis clusters counts as together with all
  of them.

An algorithmic cluster that holds no hypothesis ID is marked `new_topic` in
`clusters.csv`. These are topics outside your hypotheses.

`hypotheses/survey.json` does not exist yet. Until it does, the evaluation is
skipped for the survey.

## Troubleshooting

| Message | Meaning and fix |
|---|---|
| `--llm-url must be a full URL` | the URL is wrong, often a model name; set `--llm-url` / `VLLM_LLM_URL` to `http://<host>:<port>/v1/chat/completions` |
| `chat server check failed` | server not reachable, wrong `--llm-model` or wrong `VLLM_API_KEY` |
| `answer budget ... ran out before any answer text` | the model spends its answer budget on hidden thinking. Thinking is switched off by default (`enable_thinking=false` and `/no_think`). If the model ignores this, raise `--llm-max-tokens` (e.g. 8000) or use a model without thinking. |
| `[check] WARNING: the model still thinks` | same as above, detected at startup |
| `[retry] chat_json failed (No valid JSON ...)` | one bad answer; it is retried up to 3 times. Only `item N failed` is a real failure. Every failed answer is saved in full to `<output-dir>/llm_errors.jsonl`, with `finish_reason` (`length` = cut at the token limit). |

## Smoke test (no vLLM needed)

```bash
bash tests/smoke_test.sh
```

The test runs all three pipelines on synthetic data in both formats (400
incidents, 120 survey respondents) against a mock server. It checks only that
the code runs. It does not check the quality of the clusters.
