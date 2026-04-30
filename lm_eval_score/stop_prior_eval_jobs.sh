#!/usr/bin/env bash
# Stop prior lm_eval harness Python drivers so NPUs are not shared by overlapping runs.
# Only targets run_mmlu_vllm.py — never the caller shell (e.g. run_preset_suite.sh would match itself).
set -uo pipefail

_pat='npu_inference_profiling/lm_eval_score/run_mmlu_vllm\.py'

if ! pgrep -f "$_pat" >/dev/null 2>&1; then
  echo "stop_prior_eval_jobs: no run_mmlu_vllm.py processes."
  exit 0
fi

echo "stop_prior_eval_jobs: sending SIGTERM to prior run_mmlu_vllm.py ..."
pkill -TERM -f "$_pat" 2>/dev/null || true
sleep 5
echo "stop_prior_eval_jobs: SIGKILL stragglers..."
pkill -KILL -f "$_pat" 2>/dev/null || true
sleep 1
echo "stop_prior_eval_jobs: remaining run_mmlu_vllm.py (if any):"
pgrep -af "$_pat" 2>/dev/null || echo "(none)"
