#!/usr/bin/env bash
# Compare Qwen3.5 prefill TTFT: Triton vs PTO per-stage vs PTO megakernel.
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_benchmark_prefill_three_way.sh
#
# Optional: SEQ_LENS="512 1024 2048" WARMUP=2 REPEATS=10
# Output: OUT_DIR (default: ./_bench_prefill_<timestamp>/) with one JSONL per case:
#   triton.jsonl, pto.jsonl, pto_mega.jsonl (each line is one seq_len run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/benchmark_prefill_latency.py"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-10}"
MODEL="${MODEL:-/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/_bench_prefill_$(date +%Y%m%d%H%M%S)}"
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096}"

mkdir -p "$OUT_DIR"
for c in triton pto pto_mega; do
  : >"$OUT_DIR/${c}.jsonl"
done
RUNLOG="$OUT_DIR/run.log"

echo "[run_benchmark_prefill_three_way] writing under $OUT_DIR" | tee "$RUNLOG"

for S in $SEQ_LENS; do
  for C in pto_mega pto triton; do
    echo "[bench] seq_len=$S case=$C" | tee -a "$RUNLOG"
    python3 "$PY" --case "$C" --seq-len "$S" --warmup "$WARMUP" --repeats "$REPEATS" --model "$MODEL" \
      --device "$ASCEND_RT_VISIBLE_DEVICES" --output-jsonl "$OUT_DIR/${C}.jsonl"
  done
done

echo "[run_benchmark_prefill_three_way] done. Per-case JSONL under $OUT_DIR: triton.jsonl pto.jsonl pto_mega.jsonl" | tee -a "$RUNLOG"
