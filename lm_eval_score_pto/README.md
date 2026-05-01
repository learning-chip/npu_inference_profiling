# lm-eval with PTO megakernel (`lm_eval_score_pto`)

Run the same lm-eval bundle as [`lm_eval_score`](../lm_eval_score/README.md) (six MMLU subjects + optional GPQA Diamond + Wikitext) with [**patch_vllm_pto**](../patch_vllm_pto/) active — **megakernel by default**.

## Prerequisites

1. **Ascend NPU** — pick an idle device (`npu-smi info`), then e.g. `export ASCEND_RT_VISIBLE_DEVICES=0`.
2. **vllm-ascend PTO hook** — apply [`patch_vllm_pto/vllm_source_patch/apply_vllm_ascend_pto_hook.py`](../patch_vllm_pto/vllm_source_patch/apply_vllm_ascend_pto_hook.py) to your installed `vllm_ascend` so workers call `apply_pto_patch()` before Qwen layers bind `chunk_gated_delta_rule`.
3. **Weights / HF** — same as [`lm_eval_score/README.md`](../lm_eval_score/README.md): local checkpoints under `/scratch/model_weights/…`, optional `HF_TOKEN` for GPQA, `USE_MODELSCOPE_HUB=0` if needed.

This directory does **not** modify baseline outputs under `lm_eval_score/outputs/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_mmlu_vllm_pto.py` | Sets `VLLM_PTO_PATCH_DIR`, applies `adapt_patch(False)` + `apply_pto_patch()` in the coordinator (mirrors [`benchmark_prefill_latency.py`](../patch_vllm_pto/benchmark_prefill_latency.py)), asserts PTO wrapper is installed, then runs `lm_eval_score/run_mmlu_vllm.py` via `runpy`. |
| `--no-pto-megakernel` | Use staged PTO kernels instead of fused megakernel (stripped before delegation). |
| `run_preset_suite_pto.sh` | Four presets with **tier-split** env (8192 / 0.52 / 8 for qwen35; 4096 / 0.82 / 4 for qwen36 W8A8), matching baseline [`FULL_SUITE_REPORT.txt`](../lm_eval_score/outputs/subset_eval/run_20260430_153437/FULL_SUITE_REPORT.txt). |
| `stop_prior_pto_jobs.sh` | SIGTERM/SIGKILL `run_mmlu_vllm_pto.py`; also runs `lm_eval_score/stop_prior_eval_jobs.sh`. |
| `smoke_pto_eval.sh` | `--limit 32 --bootstrap-iters 0` on `qwen35_0_8b`; writes under `outputs/smoke/run_<stamp>/`. |
| `compare_pto_vs_baseline.py` | Diff metrics + timing vs baseline suite dir (default `run_20260430_153437`). Warns if results are bitwise-identical to baseline. |
| `summarize_suite.py` | Print a compact table from any `outputs/subset_eval/run_*` directory. |

Disable megakernel for the whole suite:

```bash
export LM_EVAL_PTO_MEGAKERNEL=0
./run_preset_suite_pto.sh
```

## Recommended workflow

1. Stop overlapping jobs: `./stop_prior_pto_jobs.sh`
2. Smoke: `./smoke_pto_eval.sh` — confirm `eval.log` contains `PTO chunk_gated_delta_rule patch is active (fused megakernel…)` from `apply.py` and that scores differ slightly from Triton baseline (not bitwise identical).
3. Full sweep: `export ASCEND_RT_VISIBLE_DEVICES=<id>` then `./run_preset_suite_pto.sh`
4. Compare vs baseline and print a table — see **Reference sweep results** for `compare_pto_vs_baseline.py` / `summarize_suite.py` commands (pass `$(cat outputs/subset_eval/LATEST_RUN_DIR.txt)` or a concrete `run_<stamp>` directory).

## Outputs

- Latest suite path: `outputs/subset_eval/LATEST_RUN_DIR.txt`
- Each run: `outputs/subset_eval/run_<stamp>/MANIFEST.txt` (records `pto_patch_dir`, `pto_megakernel`, device, tiers)
- Full-suite tee log (optional): `outputs/full_suite_console.log` when using `run_preset_suite_pto.sh | tee …`

### Reference sweep results (megakernel)

Completed four-preset suite on **Ascend NPU 1** (`ASCEND_RT_VISIBLE_DEVICES=1`), GPQA omitted (no Hub token). Artifacts:

**[`outputs/subset_eval/run_20260430_201259/`](outputs/subset_eval/run_20260430_201259)** — absolute path `/workdir/npu_inference_profiling/lm_eval_score_pto/outputs/subset_eval/run_20260430_201259`

Per preset: `eval.json`, `eval_tables.txt`, `eval.log`; harness timing keys mirror [`lm_eval_score`](../lm_eval_score/README.md).

**Speed (`timing.eval_execution_seconds` vs baseline [`lm_eval_score/outputs/subset_eval/run_20260430_153437`](../lm_eval_score/outputs/subset_eval/run_20260430_153437)):**

| Preset | Baseline (s) | PTO (s) | Δ |
|--------|--------------|---------|---|
| qwen35_0_8b | 588.56 | 556.93 | −5.4% |
| qwen35_9b | 843.31 | 794.69 | −5.8% |
| qwen36_27b_w8a8 | 2855.81 | 2496.79 | −12.6% |
| qwen36_35b_a3b_w8a8 | 2259.67 | 2084.87 | −7.7% |

**Scores:** Wikitext word PPL / bits-per-byte shifts versus that baseline run are tiny (fractional-percent scale). `compare_pto_vs_baseline.py` did **not** warn on bitwise-identical metrics (confirms numerics diverge slightly from Triton, as expected). Full per-task deltas are in the compare script output.

**Reproduce sweep** (writes a **new** `outputs/subset_eval/run_<stamp>/`; then compare using that path or `LATEST_RUN_DIR.txt`):

```bash
cd /workdir/npu_inference_profiling/lm_eval_score_pto
export ASCEND_RT_VISIBLE_DEVICES=<id>
./stop_prior_pto_jobs.sh
./run_preset_suite_pto.sh
python3 compare_pto_vs_baseline.py --pto-root "$(cat outputs/subset_eval/LATEST_RUN_DIR.txt)"
python3 summarize_suite.py "$(cat outputs/subset_eval/LATEST_RUN_DIR.txt)"
```

**Inspect the archived reference run** [`run_20260430_201259`](outputs/subset_eval/run_20260430_201259) without rerunning:

```bash
python3 compare_pto_vs_baseline.py \
  --pto-root /workdir/npu_inference_profiling/lm_eval_score_pto/outputs/subset_eval/run_20260430_201259
python3 summarize_suite.py \
  /workdir/npu_inference_profiling/lm_eval_score_pto/outputs/subset_eval/run_20260430_201259
```

## Notes

- **`eval_execution_seconds`** may improve with the megakernel on prefill-heavy phases; overall lm-eval wall time includes many non-GDN ops — treat ~10% as directional, not guaranteed per preset.
- **Wikitext** perplexity is comparable only within the same `max_model_len` tier (8192 vs 4096), same as the baseline README.
