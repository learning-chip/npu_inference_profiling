#!/usr/bin/env bash
# PTO megakernel lm-eval suite: tier-split LM_EVAL_* matches baseline FULL_SUITE_REPORT (8192/0.52/8 vs 4096/0.82/4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SKIP_STOP_PRIOR:-0}" != "1" ]]; then
  bash "${ROOT}/stop_prior_pto_jobs.sh"
fi

SKIP_GPQA_ARGS=()
if [[ "${FORCE_GPQA_DIAMOND:-0}" != "1" && -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: No HF_TOKEN / HUGGING_FACE_HUB_TOKEN — gpqa_diamond_zeroshot is gated; using --skip-gpqa-diamond." >&2
  echo "         Export HF_TOKEN or set FORCE_GPQA_DIAMOND=1 (will fail if still not authenticated)." >&2
  SKIP_GPQA_ARGS=(--skip-gpqa-diamond)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ROOT}/outputs/subset_eval"
SUITE_ROOT="${ROOT}/outputs/subset_eval/run_${STAMP}"
mkdir -p "${SUITE_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

PATCH_DIR="${ROOT}/../patch_vllm_pto"
MEGA="${LM_EVAL_PTO_MEGAKERNEL:-1}"

EXTRA_PY_ARGS=()
if [[ "${MEGA}" != "1" && "${MEGA,,}" != "true" && "${MEGA,,}" != "yes" && "${MEGA,,}" != "on" ]]; then
  EXTRA_PY_ARGS=(--no-pto-megakernel)
fi

{
  echo "suite_run_dir=${SUITE_ROOT}"
  echo "created_local=$(date -Is)"
  echo "ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-}"
  echo "pto_patch_dir=${PATCH_DIR}"
  echo "pto_megakernel=${MEGA}"
  echo "delegate_script=${ROOT}/../lm_eval_score/run_mmlu_vllm.py"
  if [[ ${#SKIP_GPQA_ARGS[@]} -eq 0 ]]; then
    echo "gpqa_diamond_zeroshot=included"
  else
    echo "gpqa_diamond_zeroshot=skipped_no_hf_token"
  fi
  echo "tasks_default=6_mmlu_subjects+gpqa_diamond_zeroshot+wikitext"
  echo "per_preset_files=eval.json eval_tables.txt eval.log"
  echo "tier_small_qwen35=8192_0.52_8"
  echo "tier_large_qwen36_w8a8=4096_0.82_4"
} >"${SUITE_ROOT}/MANIFEST.txt"

echo "${SUITE_ROOT}" >"${ROOT}/outputs/subset_eval/LATEST_RUN_DIR.txt"

PRESETS=(qwen35_0_8b qwen35_9b qwen36_27b_w8a8 qwen36_35b_a3b_w8a8)

for preset in "${PRESETS[@]}"; do
  echo "========== preset=${preset} =========="
  if [[ "${preset}" == qwen35_0_8b || "${preset}" == qwen35_9b ]]; then
    export LM_EVAL_MAX_MODEL_LEN=8192
    export LM_EVAL_GPU_MEMORY_UTILIZATION=0.52
    export LM_EVAL_MAX_BATCH_SIZE=8
  else
    export LM_EVAL_MAX_MODEL_LEN=4096
    export LM_EVAL_GPU_MEMORY_UTILIZATION=0.82
    export LM_EVAL_MAX_BATCH_SIZE=4
  fi
  mkdir -p "${SUITE_ROOT}/${preset}"
  python3 "${ROOT}/run_mmlu_vllm_pto.py" "${EXTRA_PY_ARGS[@]}" \
    --preset "${preset}" \
    "${SKIP_GPQA_ARGS[@]}" \
    --output-json "${SUITE_ROOT}/${preset}/eval.json" \
    2>&1 | tee "${SUITE_ROOT}/${preset}/eval.log"
done

echo "Done. Report tree: ${SUITE_ROOT}/"
