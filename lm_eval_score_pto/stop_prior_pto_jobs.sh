#!/usr/bin/env bash
# Stop prior lm-eval harness drivers so NPUs are not shared by overlapping runs.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_STOP="${ROOT}/../lm_eval_score/stop_prior_eval_jobs.sh"
if [[ -x "${BASE_STOP}" ]] || [[ -f "${BASE_STOP}" ]]; then
  bash "${BASE_STOP}"
fi

_pat_pto='lm_eval_score_pto/run_mmlu_vllm_pto\.py'

if ! pgrep -f "${_pat_pto}" >/dev/null 2>&1; then
  echo "stop_prior_pto_jobs: no run_mmlu_vllm_pto.py processes."
  exit 0
fi

echo "stop_prior_pto_jobs: sending SIGTERM to prior run_mmlu_vllm_pto.py ..."
pkill -TERM -f "${_pat_pto}" 2>/dev/null || true
sleep 5
echo "stop_prior_pto_jobs: SIGKILL stragglers..."
pkill -KILL -f "${_pat_pto}" 2>/dev/null || true
sleep 1
echo "stop_prior_pto_jobs: remaining run_mmlu_vllm_pto.py (if any):"
pgrep -af "${_pat_pto}" 2>/dev/null || echo "(none)"
