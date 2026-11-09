# LLM classification (approach 3)

Clustering as classification, after
[Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)
(arXiv 2410.00927). It uses no seed labels: the LLM finds the topics itself.

## How it works

1. **Generate:** the LLM reads shuffled batches of 15 documents and proposes
   topic labels. The batches run concurrently.
2. **Merge:** the LLM merges similar labels into one final list of 15 to 40
   topics, each with a short description. If there are many raw labels,
   it merges them first in chunks of 200.
3. **Classify:** the LLM assigns each document to one topic, or to
   `0 = Sonstiges`. It handles 10 documents per call.
4. **Report:** `common/report.py` writes the outputs and evaluates against
   the hypothesis clusters. It uses embeddings only for the UMAP plot.

"Sonstiges" counts as noise (`cluster_id = -1`).

## Run

Run from the folder that holds the cleaned input. See the root `README.md`
for the shared flags (format, endpoints, hypotheses file):

```bash
python <repo>/llm_classification/run.py --format incidents --embed-url ... --llm-url ...
```

`--help` lists every flag. LLM calls for about 3,500 documents with the
defaults:

| Stage | Calls |
|---|---|
| Generate | ~235 |
| Merge | 1-5 |
| Classify | ~350 |
| Hypothesis name check | 1 per hypothesis cluster |

## Resume and manual edits

Each stage saves its result in `cluster_output/<format>/llm_classification/stages/`:

| File | Content |
|---|---|
| `raw_labels.json` | raw labels per batch |
| `merged_labels.json` | final topic list |
| `classifications.jsonl` | one line per classified document |

With `--resume` (the default), the next run reuses these files. A crashed
classification continues where it stopped.

You can edit `merged_labels.json` by hand, for example to rename, delete or
add topics. To classify again with the edited list, delete
`classifications.jsonl` and run again.

Use `--no-resume` to start from scratch.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--gen-batch-size` | `15` | max documents per label generation call; fewer if they do not fit the context |
| `--gen-max-chars` | `0` | optional cap per document in generation, `0` = none |
| `--gen-sample-size` | `0` | documents used for generation, `0` = all |
| `--merge-chunk-size` | `200` | labels per pre-merge call |
| `--target-min-labels` / `--target-max-labels` | `15` / `40` | size of the final topic list |
| `--classify-batch-size` | `10` | max documents per classification call, `1` = as in the paper; fewer if they do not fit the context |
| `--classify-max-chars` | `0` | optional cap per document in classification, `0` = none |

## Outputs

See the root `README.md`, section Outputs. This approach also adds the
column `llm_failed` to `assignments.csv`. It marks documents that the LLM
could not classify; these count as "Sonstiges".
