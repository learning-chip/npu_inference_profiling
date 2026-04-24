#!/usr/bin/env bash
# Tiny Triton vs PTO eval: two ``record`` + ``compare`` (no Python subprocess.run).
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./sanity_pto_triton_eval.sh
#
# Optional env:
#   MODEL=...   OUTDIR=/tmp/pto_sanity_$$   TIMEOUT_SEC=900
#   INCLUDE_MMLU=1   MAX_MODEL_LEN=1024   WIKI_WINDOW=128   WIKI_PAGES=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/compare_pto_triton_lm_eval.py"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/_sanity_eval_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OUTDIR"

MODEL="${MODEL:-/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-768}"
WIKI_PAGES="${WIKI_PAGES:-1}"
WIKI_WINDOW="${WIKI_WINDOW:-128}"
MAX_LOGPROBS="${MAX_LOGPROBS:-131072}"
SEED="${SEED:-0}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

COMMON=(
  --model "$MODEL"
  --device "$ASCEND_RT_VISIBLE_DEVICES"
  --wiki-max-pages "$WIKI_PAGES"
  --wiki-window "$WIKI_WINDOW"
  --max-model-len "$MAX_MODEL_LEN"
  --max-logprobs "$MAX_LOGPROBS"
  --seed "$SEED"
)

MMLU_ARGS=()
if [[ "${INCLUDE_MMLU:-0}" == "1" ]]; then
  MMLU_ARGS+=(--mmlu-subjects "${MMLU_SUBJECT:-high_school_geography}")
  MMLU_ARGS+=(--mmlu-max-samples "${MMLU_MAX_SAMPLES:-2}")
  MMLU_ARGS+=(--num-fewshot "${MMLU_FEWSHOT:-1}")
else
  MMLU_ARGS+=(--skip-mmlu)
fi

run_triton() {
  echo "[sanity] Triton → $OUTDIR/triton.json (timeout ${TIMEOUT_SEC}s)" | tee "$OUTDIR/run_triton.txt"
  timeout "${TIMEOUT_SEC}" python3 "$EVAL_PY" record --backend triton \
    "${COMMON[@]}" "${MMLU_ARGS[@]}" --output "$OUTDIR/triton.json" \
    2>"$OUTDIR/triton.stderr" || {
    echo "[sanity] Triton FAILED rc=$? (see $OUTDIR/triton.stderr)" >&2
    tail -n 80 "$OUTDIR/triton.stderr" >&2 || true
    return 1
  }
  test -s "$OUTDIR/triton.json"
}

run_pto() {
  echo "[sanity] PTO → $OUTDIR/pto.json (timeout ${TIMEOUT_SEC}s)" | tee "$OUTDIR/run_pto.txt"
  timeout "${TIMEOUT_SEC}" python3 "$EVAL_PY" record --backend pto \
    "${COMMON[@]}" "${MMLU_ARGS[@]}" --output "$OUTDIR/pto.json" \
    2>"$OUTDIR/pto.stderr" || {
    echo "[sanity] PTO FAILED rc=$? (see $OUTDIR/pto.stderr)" >&2
    tail -n 80 "$OUTDIR/pto.stderr" >&2 || true
    return 1
  }
  test -s "$OUTDIR/pto.json"
}

run_triton
run_pto
python3 "$EVAL_PY" compare "$OUTDIR/triton.json" "$OUTDIR/pto.json" | tee "$OUTDIR/compare.txt"
echo "[sanity] All OK. Artifacts: $OUTDIR"
