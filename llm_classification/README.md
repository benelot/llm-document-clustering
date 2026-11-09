# LLM classification (approach 3)

Clustering as classification, after
[Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)
(arXiv 2410.00927). It uses no seed labels: the LLM finds the topics itself.

## How it works

1. **Generate:** the LLM reads shuffled batches of 15 incidents and proposes
   topic labels. The batches run concurrently.
2. **Merge:** the LLM merges similar labels into one final list of 15 to 40
   topics, each with a short description. If there are many raw labels,
   it merges them first in chunks of 200.
3. **Classify:** the LLM assigns each incident to one topic, or to
   `0 = Sonstiges`. It handles 10 incidents per call.
4. **Report:** `common/report.py` writes the outputs. It uses embeddings
   only for the UMAP plot.

"Sonstiges" counts as noise (`cluster_id = -1`).

## Run

Use the same environment variables as `naive/`. See `naive/README.md`,
section Setup. Run from the folder that holds `cleaned_incidents.json`:

```bash
python <repo>/llm_classification/run.py
```

LLM calls for about 3,500 incidents with the defaults:

| Stage | Calls |
|---|---|
| Generate | ~235 |
| Merge | 1-5 |
| Classify | ~350 |

## Resume and manual edits

Each stage saves its result in `cluster_output/llm_classification/stages/`:

| File | Content |
|---|---|
| `raw_labels.json` | raw labels per batch |
| `merged_labels.json` | final topic list |
| `classifications.jsonl` | one line per classified incident |

With `RESUME=1` (the default), the next run reuses these files. A crashed
classification continues where it stopped.

You can edit `merged_labels.json` by hand, for example to rename, delete or
add topics. To classify again with the edited list, delete
`classifications.jsonl` and run again.

Set `RESUME=0` to start from scratch.

## Parameters

| Variable | Default | Meaning |
|---|---|---|
| `LLM_CONCURRENCY` | `8` | parallel requests to vLLM |
| `GEN_BATCH_SIZE` | `15` | incidents per label generation call |
| `GEN_MAX_CHARS` | `600` | text per incident in generation |
| `GEN_SAMPLE_SIZE` | `0` | incidents used for generation, `0` = all |
| `MERGE_CHUNK_SIZE` | `200` | labels per pre-merge call |
| `TARGET_MIN_LABELS` / `TARGET_MAX_LABELS` | `15` / `40` | size of final topic list |
| `CLASSIFY_BATCH_SIZE` | `10` | incidents per classification call, `1` = as in the paper |
| `CLASSIFY_MAX_CHARS` | `1000` | text per incident in classification |
| `PLOT` | `1` | embed + UMAP plot, `0` = skip |
| `OUTPUT_DIR` | `cluster_output/llm_classification` | |
| `SEED` | `42` | shuffle seed |

## Outputs

See the root `README.md`, section Outputs. This approach also adds the
column `llm_failed` to `assignments.csv`. It marks incidents that the LLM
could not classify; these count as "Sonstiges".
