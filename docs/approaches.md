# Alternative approaches vs. the naive approach

The naive approach (`naive/`): embed the incidents, cluster them with
HDBSCAN, then let an LLM name each cluster from 8 sample incidents.

## 1. ClusterLLM, granularity part

Source: https://github.com/zhang-yu-wei/ClusterLLM/tree/main/granularity
(EMNLP 2023)

- **Problem it fixes:** HDBSCAN decides the cluster size through
  `min_cluster_size` and `min_samples`. You tune these by hand.
- **How:** sample incident pairs and ask the LLM "same topic or
  different?", with a few examples in the prompt. Then pick the number of
  clusters that agrees best with these answers.
- **Gain:** you control granularity with examples ("these two belong
  together") instead of parameters.
- The other part of ClusterLLM ("perspective") uses LLM triplet answers
  ("is B or C closer to A?") to fine-tune the embedding model, so that the
  vectors follow your notion of a topic.
- **Cost:** many LLM calls for the pairs. Fine-tuning needs a GPU.

## 2. llm_text_clustering

Source: https://github.com/v1nybarreto/llm_text_clustering

- Almost the same pipeline as the naive one: SentenceTransformer, then
  UMAP/PCA, then HDBSCAN, then GPT-4 cluster descriptions.
- Extras:
  - It reduces dimensions **before** clustering. HDBSCAN works badly
    on 1024-2560 dims.
  - It reports the silhouette score, which gives you a number to
    compare parameter settings.
  - It removes stopwords and lemmatizes the text. This is less useful
    with modern embedding models.
- **Gain:** small. The dimension reduction before HDBSCAN is worth taking.
  It is now in the naive approach as `DIM_REDUCTION=umap`.

## 3. Text Clustering via LLM (classification)

Source: https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM
(arXiv 2410.00927)

- **Problem it fixes:** the topics come from vector distances, and the LLM
  sees only 8 incidents per cluster.
- **How:** clustering becomes classification, with no embeddings:
  1. The LLM reads incidents in batches and proposes topic labels.
  2. The LLM merges similar labels into one final list.
  3. The LLM assigns each incident to one label.
- **Seed labels:** it can optionally start from a few known labels.
- **Gain:**
  - Every incident gets a readable topic.
  - There is no noise class.
  - Hypotheses can be checked directly by label.
- **Cost:** about one LLM call per incident, which is roughly 3,500 calls.
  Label quality depends on the prompt.

## 4. Vectoria

Source: https://github.com/arminpasalic/vectoria

- **Problem it fixes:** exploring and checking results. In the naive
  approach this needs separate CSV files and one static plot.
- **How:** it runs in the browser, on the local machine:
  - It embeds with multilingual-e5-small, makes a UMAP map and
    clusters with HDBSCAN.
  - It adds semantic and keyword search, and RAG questions with
    source links.
  - Clusters are named by hand or with a local LLM.
- **Gain:** you can look into clusters, find incidents similar to a given
  INC, and fix labels interactively. The data stays local.
- **Cost:** the embedding model is weaker than BGE-M3 or
  Qwen3-Embedding. It is an exploration tool, not a batch pipeline.

## Recommendation

- Best fit for the goal: approach 3.
- Cheapest quick gain: dimension reduction before HDBSCAN (approach 2).
