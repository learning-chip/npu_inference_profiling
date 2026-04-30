#!/usr/bin/env bash
# Default bundle: 6 MMLU subjects + gpqa_diamond_zeroshot + wikitext (see run_mmlu_vllm.py).
# Writes a fresh timestamped directory under outputs/subset_eval/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SKIP_STOP_PRIOR:-0}" != "1" ]]; then
  bash "${ROOT}/stop_prior_eval_jobs.sh"
fi

SKIP_GPQA_ARGS=()
if [[ "${FORCE_GPQA_DIAMOND:-0}" != "1" && -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: No HF_TOKEN / HUGGING_FACE_HUB_TOKEN — gpqa_diamond_zeroshot is gated; using --skip-gpqa-diamond." >&2
  echo "         Export HF_TOKEN or set FORCE_GPQA_DIAMOND=1 (will fail if still not authenticated)." >&2
  SKIP_GPQA_ARGS=(--skip-gpqa-diamond)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${ROOT}/outputs/subset_eval/run_${STAMP}"
mkdir -p "${SUITE_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export LM_EVAL_GPU_MEMORY_UTILIZATION="${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.52}"
export LM_EVAL_MAX_BATCH_SIZE="${LM_EVAL_MAX_BATCH_SIZE:-8}"

PRESETS=(qwen35_0_8b qwen35_9b qwen36_27b_w8a8 qwen36_35b_a3b_w8a8)

{
  echo "suite_run_dir=${SUITE_ROOT}"
  echo "created_local=$(date -Is)"
  echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-}"
  echo "lm_eval_gpu_memory_utilization=${LM_EVAL_GPU_MEMORY_UTILIZATION:-}"
  echo "lm_eval_max_batch_size=${LM_EVAL_MAX_BATCH_SIZE:-}"
  if [[ ${#SKIP_GPQA_ARGS[@]} -eq 0 ]]; then
    echo "gpqa_diamond_zeroshot=included"
  else
    echo "gpqa_diamond_zeroshot=skipped_no_hf_token"
  fi
  echo "tasks_default=6_mmlu_subjects+gpqa_diamond_zeroshot+wikitext"
  echo "per_preset_files=eval.json eval_tables.txt eval.log"
} >"${SUITE_ROOT}/MANIFEST.txt"

echo "${SUITE_ROOT}" >"${ROOT}/outputs/subset_eval/LATEST_RUN_DIR.txt"

for preset in "${PRESETS[@]}"; do
  echo "========== preset=${preset} =========="
  mkdir -p "${SUITE_ROOT}/${preset}"
  python3 "${ROOT}/run_mmlu_vllm.py" \
    --preset "${preset}" \
    "${SKIP_GPQA_ARGS[@]}" \
    --output-json "${SUITE_ROOT}/${preset}/eval.json" \
    2>&1 | tee "${SUITE_ROOT}/${preset}/eval.log"
done

echo "Done. Report tree: ${SUITE_ROOT}/"
