#!/usr/bin/env bash
# Run Triton then PTO ``compare_prefill_next_token.py record`` in two separate
# Python invocations (no nested subprocess inside Python), then ``compare``.
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_compare_prefill.sh
#
# Optional: SEQ_LEN=64 NUM_GEN=5 OUTDIR=/tmp/my_cmp ./run_compare_prefill.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/compare_prefill_next_token.py"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/_prefill_cmp_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OUTDIR"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_GEN="${NUM_GEN:-11}"
MODEL="${MODEL:-/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/}"
MAX_LP="${MAX_LP:-300000}"

COMMON=(--model "$MODEL" --device "$ASCEND_RT_VISIBLE_DEVICES" --seq-len "$SEQ_LEN" --num-generated "$NUM_GEN" --max-logprobs "$MAX_LP")

echo "[run_compare_prefill] Triton → $OUTDIR/triton.npz"
python3 "$PY" record --backend triton --output "$OUTDIR/triton.npz" "${COMMON[@]}" \
  >"$OUTDIR/triton.meta.json" 2>"$OUTDIR/triton.log"

echo "[run_compare_prefill] PTO → $OUTDIR/pto.npz"
python3 "$PY" record --backend pto --output "$OUTDIR/pto.npz" "${COMMON[@]}" \
  >"$OUTDIR/pto.meta.json" 2>"$OUTDIR/pto.log"

echo "[run_compare_prefill] compare"
python3 "$PY" compare "$OUTDIR/triton.npz" "$OUTDIR/pto.npz" | tee "$OUTDIR/compare.txt"
echo "[run_compare_prefill] OK. Artifacts in $OUTDIR"
