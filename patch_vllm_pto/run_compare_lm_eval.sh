#!/usr/bin/env bash
# Two separate ``python3 compare_pto_triton_lm_eval.py record`` invocations + ``compare``.
# No Python ``subprocess.run`` (vLLM already spawns workers).
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_compare_lm_eval.sh
#
# Optional: OUTDIR=/tmp/lm WIKI_PAGES=1 WIKI_WINDOW=128 MAX_MODEL_LEN=768 SKIP_MMLU=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/compare_pto_triton_lm_eval.py"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/_lm_eval_cmp_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OUTDIR"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/}"
WIKI_PAGES="${WIKI_PAGES:-1}"
WIKI_WINDOW="${WIKI_WINDOW:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-768}"
MAX_LP="${MAX_LP:-131072}"
SKIP_MMLU="${SKIP_MMLU:-1}"

COMMON=(
  --model "$MODEL"
  --device "$ASCEND_RT_VISIBLE_DEVICES"
  --wiki-max-pages "$WIKI_PAGES"
  --wiki-window "$WIKI_WINDOW"
  --max-model-len "$MAX_MODEL_LEN"
  --max-logprobs "$MAX_LP"
)

SKIP_ARGS=()
if [[ "$SKIP_MMLU" == "1" ]]; then
  SKIP_ARGS+=(--skip-mmlu)
fi

echo "[run_compare_lm_eval] Triton → $OUTDIR/triton.json"
python3 "$PY" record --backend triton "${COMMON[@]}" "${SKIP_ARGS[@]}" --output "$OUTDIR/triton.json" \
  2>"$OUTDIR/triton.stderr" || {
  echo "Triton record failed; tail:" >&2
  tail -n 60 "$OUTDIR/triton.stderr" >&2 || true
  exit 1
}

echo "[run_compare_lm_eval] PTO → $OUTDIR/pto.json"
python3 "$PY" record --backend pto "${COMMON[@]}" "${SKIP_ARGS[@]}" --output "$OUTDIR/pto.json" \
  2>"$OUTDIR/pto.stderr" || {
  echo "PTO record failed; tail:" >&2
  tail -n 60 "$OUTDIR/pto.stderr" >&2 || true
  exit 1
}

echo "[run_compare_lm_eval] compare"
python3 "$PY" compare "$OUTDIR/triton.json" "$OUTDIR/pto.json" | tee "$OUTDIR/compare.txt"
echo "[run_compare_lm_eval] OK. Artifacts: $OUTDIR"
