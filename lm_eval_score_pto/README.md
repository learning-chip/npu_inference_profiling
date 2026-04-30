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
4. Compare:

```bash
python3 compare_pto_vs_baseline.py \
  --pto-root "$(cat outputs/subset_eval/LATEST_RUN_DIR.txt)"
```

5. Optional table:

```bash
python3 summarize_suite.py "$(cat outputs/subset_eval/LATEST_RUN_DIR.txt)"
```

## Outputs

- Latest suite path: `outputs/subset_eval/LATEST_RUN_DIR.txt`
- Each run: `outputs/subset_eval/run_<stamp>/MANIFEST.txt` (records `pto_patch_dir`, `pto_megakernel`, device, tiers)

## Notes

- **`eval_execution_seconds`** may improve with the megakernel on prefill-heavy phases; overall lm-eval wall time includes many non-GDN ops — treat ~10% as directional, not guaranteed per preset.
- **Wikitext** perplexity is comparable only within the same `max_model_len` tier (8192 vs 4096), same as the baseline README.
