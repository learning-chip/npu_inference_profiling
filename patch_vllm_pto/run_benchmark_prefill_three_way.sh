#!/usr/bin/env bash
# Compare Qwen3.5 prefill TTFT: Triton vs PTO per-stage vs PTO megakernel.
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_benchmark_prefill_three_way.sh
#
# Optional: SEQ_LENS="512 1024 2048" WARMUP=2 REPEATS=10
# SEQ_LENS are swept in one Python process per backend (one vLLM load per case).
# Models: space-separated "LABEL:SNAPSHOT_DIR" pairs in BENCHMARK_MODEL_SPECS (default: 2B, 0.8B, Qwen3.5-4B, Qwen3.5-9B — the latter two use ``linear_num_value_heads`` > ``linear_num_key_heads`` and need the ``dynamic_bsnd_groupvalue`` / ``pto_mega_kernel_groupvalue`` path in ``pto_chunk_gated_delta_rule.py``).
# Output: OUT_DIR/<LABEL>/{triton,pto,pto_mega}.jsonl — each JSON line includes model_label and model path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/benchmark_prefill_latency.py"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-10}"
_SNAP_0_8B="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/"
_SNAP_2B="/scratch/model_weights/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/"
_SNAP_4B="/scratch/model_weights/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/"
_SNAP_9B="/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/"
_DEFAULT_SPECS="2B:$_SNAP_2B 0.8B:$_SNAP_0_8B 4B:$_SNAP_4B 9B:$_SNAP_9B"
BENCHMARK_MODEL_SPECS="${BENCHMARK_MODEL_SPECS:-$_DEFAULT_SPECS}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/_bench_prefill_$(date +%Y%m%d%H%M%S)}"
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096 8192 16384 32768 65536}"

mkdir -p "$OUT_DIR"
RUNLOG="$OUT_DIR/run.log"

echo "[run_benchmark_prefill_three_way] writing under $OUT_DIR" | tee "$RUNLOG"
echo "[run_benchmark_prefill_three_way] models: $BENCHMARK_MODEL_SPECS" | tee -a "$RUNLOG"

for SPEC in $BENCHMARK_MODEL_SPECS; do
  LABEL="${SPEC%%:*}"
  MPATH="${SPEC#*:}"
  SUB="$OUT_DIR/$LABEL"
  mkdir -p "$SUB"
  for c in triton pto pto_mega; do
    : >"$SUB/${c}.jsonl"
  done
  echo "[run_benchmark_prefill_three_way] model_label=$LABEL path=$MPATH" | tee -a "$RUNLOG"
  # One Python job per backend: reuse one vLLM session across SEQ_LENS (init is slow).
  for C in pto_mega pto triton; do
    echo "[bench] model_label=$LABEL case=$C seq_lens=$SEQ_LENS" | tee -a "$RUNLOG"
    python3 "$PY" --case "$C" --model-label "$LABEL" --seq-len $SEQ_LENS --warmup "$WARMUP" --repeats "$REPEATS" \
      --model "$MPATH" --device "$ASCEND_RT_VISIBLE_DEVICES" --output-jsonl "$SUB/${C}.jsonl"
  done
done

echo "[run_benchmark_prefill_three_way] done. Per-model dirs under $OUT_DIR: <LABEL>/{triton,pto,pto_mega}.jsonl" | tee -a "$RUNLOG"
