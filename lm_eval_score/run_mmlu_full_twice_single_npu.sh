#!/usr/bin/env bash
# Two identical eval passes on one NPU (tensor_parallel_size=1), same seed — compare metrics.
# Default tasks are lightweight (two MMLU subjects). Override with TASKS=...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SKIP_STOP_PRIOR:-0}" != "1" ]]; then
  bash "${ROOT}/stop_prior_eval_jobs.sh"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/outputs/determinism/${STAMP}"
mkdir -p "${OUT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
SEED="${SEED:-1234}"
TASKS="${TASKS:-mmlu_astronomy,mmlu_high_school_mathematics}"

{
  echo "determinism_run_dir=${OUT}"
  echo "tasks=${TASKS}"
  echo "seed=${SEED}"
} >"${OUT}/MANIFEST.txt"
echo "${OUT}" >"${ROOT}/outputs/determinism/LATEST_RUN_DIR.txt"

for run in 1 2; do
  echo "========== determinism run ${run} =========="
  python3 "${ROOT}/run_mmlu_vllm.py" \
    --tasks "${TASKS}" \
    --tensor-parallel-size 1 \
    --seed "${SEED}" \
    --output-json "${OUT}/run_${run}.json" \
    2>&1 | tee "${OUT}/run_${run}.log"
done

python3 "${ROOT}/compare_mmlu_runs.py" \
  "${OUT}/run_1.json" \
  "${OUT}/run_2.json" \
  | tee "${OUT}/compare.txt"
