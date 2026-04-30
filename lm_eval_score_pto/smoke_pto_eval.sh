#!/usr/bin/env bash
# Quick PTO sanity: tiny limit, megakernel default. Inspect eval.log for apply.py warning and wrapper assertion.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/outputs/smoke/run_${STAMP}"
mkdir -p "${OUT}"

echo "Smoke output: ${OUT}"
python3 "${ROOT}/run_mmlu_vllm_pto.py" \
  --preset qwen35_0_8b \
  --skip-gpqa-diamond \
  --limit 32 \
  --bootstrap-iters 0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.52 \
  --max-batch-size 8 \
  --output-json "${OUT}/eval.json" \
  2>&1 | tee "${OUT}/eval.log"

echo "Done. See ${OUT}/eval.json and eval.log"
