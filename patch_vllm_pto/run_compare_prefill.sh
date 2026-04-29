#!/usr/bin/env bash
# Run Triton then PTO ``compare_prefill_next_token.py record`` in two separate
# Python invocations (no nested subprocess inside Python), then ``compare``.
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_compare_prefill.sh
#
# Optional: SEQ_LEN=64 NUM_GEN=5 OUTDIR=/tmp/my_cmp ./run_compare_prefill.sh
# Models: space-separated "LABEL:SNAPSHOT_DIR" pairs in BENCHMARK_MODEL_SPECS
# (default: 2B, 0.8B, Qwen3.5-4B, Qwen3.5-9B, **Qwen3.6-27B-w8a8** — GQA needs groupvalue PTO; W8A8 needs
# **`export BENCH_QUANTIZATION=ascend`** alongside **``VLLM_PTO_PATCH_DIR``**).
# Megakernel record+compare runs by default; skip with COMPARE_MEGA=0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/compare_prefill_next_token.py"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/_prefill_cmp_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OUTDIR"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_GEN="${NUM_GEN:-11}"
MAX_LP="${MAX_LP:-300000}"
_SNAP_0_8B="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/"
_SNAP_2B="/scratch/model_weights/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/"
_SNAP_4B="/scratch/model_weights/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/"
_SNAP_9B="/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/"
_SNAP_27B_W8A8="/scratch/model_weights/Qwen3.6-27B-w8a8/"
_DEFAULT_SPECS="2B:$_SNAP_2B 0.8B:$_SNAP_0_8B 4B:$_SNAP_4B 9B:$_SNAP_9B 27B-w8a8:$_SNAP_27B_W8A8"
BENCHMARK_MODEL_SPECS="${BENCHMARK_MODEL_SPECS:-$_DEFAULT_SPECS}"
# Forward to ``compare_prefill_next_token.py record`` (needed for msmodelslim W8A8 when **27B-w8a8** is in specs).
BENCH_QUANTIZATION="${BENCH_QUANTIZATION:-}"

echo "[run_compare_prefill] models: $BENCHMARK_MODEL_SPECS"
echo "[run_compare_prefill] BENCH_QUANTIZATION=${BENCH_QUANTIZATION:-<unset>}"

for SPEC in $BENCHMARK_MODEL_SPECS; do
  LABEL="${SPEC%%:*}"
  MPATH="${SPEC#*:}"
  SUB="$OUTDIR/$LABEL"
  mkdir -p "$SUB"

  COMMON=(--model "$MPATH" --device "$ASCEND_RT_VISIBLE_DEVICES" --seq-len "$SEQ_LEN" --num-generated "$NUM_GEN" --max-logprobs "$MAX_LP")
  if [[ -n "${BENCH_QUANTIZATION}" ]]; then
    COMMON+=(--quantization "${BENCH_QUANTIZATION}")
  fi

  echo "[run_compare_prefill] model_label=$LABEL PTO → $SUB/pto.npz"
  python3 "$PY" record --backend pto --output "$SUB/pto.npz" "${COMMON[@]}" \
    >"$SUB/pto.meta.json" 2>"$SUB/pto.log"

  echo "[run_compare_prefill] model_label=$LABEL Triton → $SUB/triton.npz"
  python3 "$PY" record --backend triton --output "$SUB/triton.npz" "${COMMON[@]}" \
    >"$SUB/triton.meta.json" 2>"$SUB/triton.log"

  echo "[run_compare_prefill] model_label=$LABEL compare (Triton vs PTO per-stage)"
  python3 "$PY" compare "$SUB/triton.npz" "$SUB/pto.npz" | tee "$SUB/compare.txt"

  echo "[run_compare_prefill] model_label=$LABEL PTO megakernel → $SUB/pto_mega.npz"
  python3 "$PY" record --backend pto_mega --output "$SUB/pto_mega.npz" "${COMMON[@]}" \
    >"$SUB/pto_mega.meta.json" 2>"$SUB/pto_mega.log"
  echo "[run_compare_prefill] model_label=$LABEL compare (Triton vs PTO megakernel)"
  python3 "$PY" compare "$SUB/triton.npz" "$SUB/pto_mega.npz" | tee "$SUB/compare_mega.txt"
done

echo "[run_compare_prefill] OK. Artifacts under $OUTDIR/<LABEL>/ (e.g. 2B/, 0.8B/, …, 27B-w8a8/)"
