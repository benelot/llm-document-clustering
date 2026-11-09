# ClusterLLM granularity (approach 1)

The granularity part of
[ClusterLLM](https://github.com/zhang-yu-wei/ClusterLLM) (EMNLP 2023).
The LLM decides how fine the clusters are, instead of HDBSCAN parameters.
The triplet fine-tuning part ("perspective") is not included.

## How it works

1. **Embed** all incidents, with the same API and cache as the other
   approaches. Then reduce the dimensions (default: UMAP to 10 dims).
2. **Tree:** build a full agglomerative (Ward) tree.
3. **Pairs:** for every merge step from `K_MAX` down to `K_MIN` clusters,
   take `PAIRS_PER_MERGE` pairs. Each pair has one incident from each of
   the two merged clusters. The LLM answers: same problem category?
4. **Choose k:** for each k, count how many answers agree with cutting the
   tree at k:
   - merges above k should be "same";
   - merges at or below k should be "different".

   The best k wins. See `k_scores.png`.
5. **Cut and name:** cut the tree at k. The LLM names each cluster from its
   8 most central incidents.

There is no noise: every incident gets a cluster.

## Run

Use the same environment variables as `naive/`. Run from the folder that
holds `cleaned_incidents.json`:

```bash
python <repo>/clusterllm_granularity/run.py
```

With the defaults, it makes 90 merges x 5 pairs = 450 pair calls, plus one
naming call per cluster.

## Parameters

| Variable | Default | Meaning |
|---|---|---|
| `LLM_CONCURRENCY` | `8` | parallel requests to vLLM |
| `DIM_REDUCTION` | `umap` | `none`, `pca` or `umap`, before building the tree |
| `DIMRED_N_COMPONENTS` | `10` | target dims |
| `K_MIN` / `K_MAX` | `10` / `100` | candidate range for the number of clusters |
| `PAIRS_PER_MERGE` | `5` | LLM questions per merge step |
| `PAIR_MAX_CHARS` | `1000` | text per incident in a pair question |
| `GRANULARITY_QUESTION` | see `run.py` | the "same category?" question; this defines the granularity |
| `FORCE_K` | `0` | `>0` skips the pair stage and uses this k |
| `NAME_SAMPLES` | `8` | central incidents shown for naming |
| `OUTPUT_DIR` | `cluster_output/clusterllm_granularity` | |

Pair answers are saved in `stages/pairs.csv`. With `RESUME=1` (the
default), the next run reuses them.

To change the granularity, change `GRANULARITY_QUESTION`. Then run again
with `RESUME=0`.

## Outputs

See the root `README.md`, section Outputs. This approach also writes:

| File | Content |
|---|---|
| `k_scores.csv` | agreement per k |
| `k_scores.png` | agreement curve, LLM "same" share per merge, chosen k |
