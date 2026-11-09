#!/usr/bin/env bash
# End-to-end smoke test of all pipelines against the mock vLLM server,
# for both input formats (incidents and CIRS survey). Synthetic data only.
# Usage: bash tests/smoke_test.sh [workdir]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
PORT=8099
mkdir -p "$WORK"
cd "$WORK"

python "$REPO/tests/mock_vllm.py" $PORT & MOCK=$!
trap 'kill $MOCK' EXIT
sleep 1

export VLLM_EMBED_URL="http://127.0.0.1:$PORT/v1/embeddings"
export VLLM_LLM_URL="http://127.0.0.1:$PORT/v1/chat/completions"
export EMBED_BACKEND=api LLM_RETRIES=2 RESUME=0
export K_MIN=3 K_MAX=20 PAIRS_PER_MERGE=3 TARGET_MIN_LABELS=4 TARGET_MAX_LABELS=10
export HDBSCAN_MIN_CLUSTER_SIZE=5

echo "=== Format 1: incidents ==="
python "$REPO/tests/make_synthetic_data.py" cleaned_incidents.json 400
export DATA_PROFILE=incidents CLEANED_FILE=cleaned_incidents.json
OUTPUT_DIR=cluster_output/incidents/naive DIM_REDUCTION=umap python "$REPO/naive/cluster_incidents.py"
OUTPUT_DIR=cluster_output/incidents/llm_classification python "$REPO/llm_classification/run.py"
OUTPUT_DIR=cluster_output/incidents/clusterllm_granularity python "$REPO/clusterllm_granularity/run.py"

echo "=== Format 2: CIRS survey ==="
python "$REPO/tests/make_synthetic_cirs.py" cirs.xlsx 120
python "$REPO/naive/preprocess_cirs.py" cirs.xlsx cleaned_cirs.csv
export DATA_PROFILE=cirs CLEANED_FILE=cleaned_cirs.json
OUTPUT_DIR=cluster_output/cirs/naive DIM_REDUCTION=umap python "$REPO/naive/cluster_incidents.py"
OUTPUT_DIR=cluster_output/cirs/llm_classification python "$REPO/llm_classification/run.py"
OUTPUT_DIR=cluster_output/cirs/clusterllm_granularity python "$REPO/clusterllm_granularity/run.py"

echo "Smoke test OK. Outputs in $WORK/cluster_output"
