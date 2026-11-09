#!/usr/bin/env bash
# End-to-end smoke test of both approaches against the mock vLLM server.
# Uses synthetic data only. Usage: bash tests/smoke_test.sh [workdir]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
PORT=8099
mkdir -p "$WORK"
cd "$WORK"

python "$REPO/tests/make_synthetic_data.py" cleaned_incidents.json 400
python "$REPO/tests/mock_vllm.py" $PORT & MOCK=$!
trap 'kill $MOCK' EXIT
sleep 1

export VLLM_EMBED_URL="http://127.0.0.1:$PORT/v1/embeddings"
export VLLM_LLM_URL="http://127.0.0.1:$PORT/v1/chat/completions"
export LLM_RETRIES=2 RESUME=0
export K_MIN=3 K_MAX=20 PAIRS_PER_MERGE=3 TARGET_MIN_LABELS=4 TARGET_MAX_LABELS=10

python "$REPO/llm_classification/run.py"
python "$REPO/clusterllm_granularity/run.py"
echo "Smoke test OK. Outputs in $WORK/cluster_output"
