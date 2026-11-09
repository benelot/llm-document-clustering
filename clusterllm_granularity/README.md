# ClusterLLM granularity (approach 1)

The granularity part of
[ClusterLLM](https://github.com/zhang-yu-wei/ClusterLLM) (EMNLP 2023).
The LLM decides how fine the clusters are, instead of HDBSCAN parameters.
The triplet fine-tuning part ("perspective") is not included.

## How it works

1. **Embed** all documents, with the same API and cache as the other
   approaches. Then reduce the dimensions (default: UMAP to 10 dims).
2. **Tree:** build a full agglomerative (Ward) tree.
3. **Pairs:** for every merge step from `--k-max` down to `--k-min` clusters,
   take `--pairs-per-merge` pairs. Each pair has one document from each of
   the two merged clusters. The LLM answers: same problem category?
4. **Choose k:** for each k, count how many answers agree with cutting the
   tree at k:
   - merges above k should be "same";
   - merges at or below k should be "different".

   The best k wins. See `k_scores.png`.
5. **Cut and name:** cut the tree at k. The LLM names each cluster from its
   8 most central documents.

There is no noise: every document gets a cluster.

## Run

Run from the folder that holds the cleaned input. See the root `README.md`
for the shared flags (format, endpoints, hypotheses file):

```bash
python <repo>/clusterllm_granularity/run.py --format incidents --embed-url ... --llm-url ...
```

With the defaults, it makes 90 merges x 5 pairs = 450 pair calls, plus one
naming call per cluster. `--help` lists every flag.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--dim-reduction` | `umap` | `none`, `pca` or `umap`, before building the tree |
| `--dimred-n-components` | `10` | target dims |
| `--k-min` / `--k-max` | `10` / `100` | candidate range for the number of clusters |
| `--pairs-per-merge` | `5` | LLM questions per merge step |
| `--pair-max-chars` | `1000` | text per document in a pair question |
| `--granularity-question` | depends on `--format` | the "same category?" question; this defines the granularity |
| `--force-k` | `0` | `>0` skips the pair stage and uses this k |
| `--name-samples` | `8` | central documents shown for naming |

The reduced vectors and pair answers are saved in `stages/`. With
`--resume` (the default), the next run reuses them.

To change the granularity, change `--granularity-question`, then run again
with `--no-resume`.

## Outputs

See the root `README.md`, section Outputs. This approach also writes:

| File | Content |
|---|---|
| `k_scores.csv` | agreement per k |
| `k_scores.png` | agreement curve, LLM "same" share per merge, chosen k |
