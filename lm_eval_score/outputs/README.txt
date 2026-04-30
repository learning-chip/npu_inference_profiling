Directory layout (lm-eval reports):

- `subset_eval/run_<timestamp>/` — one **fresh** multi-model suite per invocation of `run_preset_suite.sh`.
  - `MANIFEST.txt` — paths and task bundle description
  - `<preset>/eval.json` — metrics, timing, `mmlu_coverage`
  - `<preset>/eval_tables.txt` — markdown tables + timing line
  - `<preset>/eval.log` — full stdout/stderr
- `subset_eval/LATEST_RUN_DIR.txt` — absolute path to the most recent suite directory
- `determinism/<timestamp>/` — outputs from `run_mmlu_full_twice_single_npu.sh` (two runs + compare)

Older flat files (`*_lm_eval.json` in this folder root) may be legacy; prefer `subset_eval/run_*`.

Before launching a new suite manually, run `/workdir/npu_inference_profiling/lm_eval_score/stop_prior_eval_jobs.sh` so stray **`run_mmlu_vllm.py`** processes release the NPUs (preset/determinism scripts invoke it by default). It does **not** kill `run_preset_suite.sh` itself.
