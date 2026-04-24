#!/usr/bin/env bash
# Compare Qwen3.5 prefill TTFT: Triton vs PTO per-stage vs PTO megakernel.
#
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   ./run_benchmark_prefill_three_way.sh
#
# Optional: SEQ_LENS="512 1024 2048" WARMUP=2 REPEATS=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/benchmark_prefill_latency.py"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-10}"
MODEL="${MODEL:-/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/}"
OUT="${OUT:-${SCRIPT_DIR}/_bench_prefill_$(date +%Y%m%d%H%M%S).jsonl}"
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096}"

: >"$OUT"
echo "[run_benchmark_prefill_three_way] writing $OUT"

for S in $SEQ_LENS; do
  for C in triton pto pto_mega; do
    echo "[bench] seq_len=$S case=$C" | tee -a "$OUT.runlog"
    python3 "$PY" --case "$C" --seq-len "$S" --warmup "$WARMUP" --repeats "$REPEATS" --model "$MODEL" \
      --device "$ASCEND_RT_VISIBLE_DEVICES" --output-jsonl "$OUT"
  done
done

echo "[run_benchmark_prefill_three_way] done. Parse $OUT (one JSON object per line)."
