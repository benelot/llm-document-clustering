#!/usr/bin/env bash
# End-to-end smoke test of all pipelines against the mock vLLM server,
# for both input formats (incidents and survey). Synthetic data only.
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

API=(--embed-url "http://127.0.0.1:$PORT/v1/embeddings"
     --llm-url "http://127.0.0.1:$PORT/v1/chat/completions"
     --llm-retries 2 --no-resume --llm-max-tokens 400)
NAIVE=(--dim-reduction umap --hdbscan-min-cluster-size 5)
CLASSIFY=(--target-min-labels 4 --target-max-labels 10)
GRAN=(--k-min 3 --k-max 20 --pairs-per-merge 3)

echo "=== Format 1: incidents ==="
python "$REPO/tests/make_synthetic_data.py" cleaned_incidents.json 400
python "$REPO/naive/cluster_incidents.py"           --format incidents "${API[@]}" "${NAIVE[@]}"
python "$REPO/llm_classification/run.py"            --format incidents "${API[@]}" "${CLASSIFY[@]}"
python "$REPO/clusterllm_granularity/run.py"        --format incidents "${API[@]}" "${GRAN[@]}"

echo "=== Format 2: survey ==="
python "$REPO/tests/make_synthetic_survey.py" survey.xlsx 120
python "$REPO/naive/preprocess_survey.py" --input survey.xlsx
# synthetic hypothesis clusters: respondents grouped by the topic of their "Fix 1" answer
python - <<'PY'
import json, pandas as pd
df = pd.read_excel("survey.xlsx")
hypotheses = {}
for kw, cat in [("Drucker", "Drucken"), ("Warnmeldung", "Warnmeldung"), ("Login", "Login")]:
    ids = df.loc[df["Fix 1"].astype(str).str.contains(kw), "RespondentID"].astype(str).tolist()
    hypotheses[cat] = ids[:6]
json.dump(hypotheses, open("hypotheses_survey.json", "w"), indent=2)
PY
python "$REPO/naive/cluster_incidents.py"           --format survey --hypotheses-file hypotheses_survey.json "${API[@]}" "${NAIVE[@]}"
python "$REPO/llm_classification/run.py"            --format survey --hypotheses-file hypotheses_survey.json "${API[@]}" "${CLASSIFY[@]}"
python "$REPO/clusterllm_granularity/run.py"        --format survey --hypotheses-file hypotheses_survey.json "${API[@]}" "${GRAN[@]}"

echo "Smoke test OK. Outputs in $WORK/cluster_output"
