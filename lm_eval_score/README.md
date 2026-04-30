# lm-eval on vLLM-Ascend (`lm_eval_score`)

Evaluate checkpoints with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) using the offline **`vllm`** backend (`lm_eval.simple_evaluate(..., model="vllm")`, same idea as `vllm-ascend/tests/e2e/singlecard/test_eager_mode_acc.py`).

**Script:** `/workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py`  
**Artifacts:** `/workdir/npu_inference_profiling/lm_eval_score/outputs/` (see `outputs/README.txt` for layout)

---

## Prerequisites

- **Hardware:** Ascend NPU(s); driver stack and `vllm-ascend` installed in the Python environment you use below.
- **Weights:** Local snapshots under `/scratch/model_weights/...` (see presets) must exist on your machine.
- **Data:** Tasks download datasets via Hugging Face unless you configure offline caches. **GPQA Diamond** (`gpqa_diamond_zeroshot`) uses the gated Hub dataset `Idavidrein/gpqa` — export **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`) with read access, or `run_preset_suite.sh` will omit GPQA with a warning. Set **`FORCE_GPQA_DIAMOND=1`** to force inclusion (evaluation fails if still unauthenticated).

If vLLM was built with ModelScope hub overrides, set `USE_MODELSCOPE_HUB=0` when evaluating so lm-eval can reach Hugging Face ([vLLM-Ascend lm-eval notes](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/evaluation/using_lm_eval.md)).

---

## Environment (copy before any run)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

Optional:

```bash
export USE_MODELSCOPE_HUB=0
# export HF_ENDPOINT=https://hf-mirror.com
```

```bash
chmod +x /workdir/npu_inference_profiling/lm_eval_score/run_preset_suite.sh
chmod +x /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_full_twice_single_npu.sh
chmod +x /workdir/npu_inference_profiling/lm_eval_score/stop_prior_eval_jobs.sh
```

**Before starting a new eval**, stop overlapping **Python** harness processes (`run_mmlu_vllm.py` only — suite scripts are not killed so a newly started `run_preset_suite.sh` cannot terminate itself):

```bash
/workdir/npu_inference_profiling/lm_eval_score/stop_prior_eval_jobs.sh
```

---

## Default benchmark bundle (fast MMLU slice)

Unless you pass **`--tasks`** or **`--full-mmlu`**, one Python invocation loads the model **once**, then runs:

**Six MMLU subjects** (representative subset):

- `mmlu_astronomy`, `mmlu_high_school_mathematics`, `mmlu_college_biology`
- `mmlu_high_school_world_history`, `mmlu_professional_law`, `mmlu_philosophy`

**Plus:**

| Task | Metrics |
|------|---------|
| `gpqa_diamond_zeroshot` | GPQA Diamond, multiple-choice accuracy |
| `wikitext` | Wikitext-2 validation perplexity (`word_perplexity`, `byte_perplexity`, `bits_per_byte`) |

**Full MMLU** (slow — entire `mmlu` task group):

```bash
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --preset qwen35_0_8b \
  --full-mmlu \
  --output-json /tmp/full_mmlu_run.json
```

**Custom tasks:**

```bash
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --preset qwen35_0_8b \
  --tasks mmlu_astronomy,gpqa_diamond_zeroshot \
  --output-json /tmp/custom.json
```

JSON field **`mmlu_coverage`**: `mmlu_subset_6` | `full_mmlu_group` | `custom_tasks`.

---

## Model presets (`--preset`)

| `--preset` | Checkpoint |
|------------|------------|
| `qwen35_0_8b` | `/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/` |
| `qwen35_9b` | `/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `qwen36_27b_w8a8` | `/scratch/model_weights/Qwen3.6-27B-w8a8` |
| `qwen36_35b_a3b_w8a8` | `/scratch/model_weights/Qwen3.6-35B-A3B-w8a8` |

Dense presets omit Ascend quantization; W8A8 presets use `quantization=ascend` and MoE expert parallel where applicable.

---

## Commands

### 1. One preset — default bundle

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
mkdir -p /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs

python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --preset qwen36_35b_a3b_w8a8 \
  --output-json /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs/qwen36_35b.json
```

Writes `qwen36_35b.json`, `qwen36_35b_tables.txt`.

### 2. All four presets — **organized suite** (recommended)

Creates **`outputs/subset_eval/run_<timestamp>/`** with `MANIFEST.txt` and one subdirectory per preset (`eval.json`, `eval_tables.txt`, `eval.log`):

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
/workdir/npu_inference_profiling/lm_eval_score/run_preset_suite.sh
```

This invokes **`stop_prior_eval_jobs.sh`** first (unless `SKIP_STOP_PRIOR=1`) so a previous background eval does not share the device.

Latest suite path:

```bash
cat /workdir/npu_inference_profiling/lm_eval_score/outputs/subset_eval/LATEST_RUN_DIR.txt
```

### 3. Default checkpoint path (no `--preset`)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --output-json /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs/default_model.json
```

### 4. Ad-hoc dense checkpoint

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --model /scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/ \
  --no-quantization \
  --no-expert-parallel \
  --output-json /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs/custom_dense.json
```

### 5. Smoke run (`--limit`)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --preset qwen35_0_8b \
  --limit 32 \
  --bootstrap-iters 0 \
  --output-json /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs/smoke.json
```

### 6. Two NPUs (`TP=2`)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 /workdir/npu_inference_profiling/lm_eval_score/run_mmlu_vllm.py \
  --preset qwen36_35b_a3b_w8a8 \
  --tensor-parallel-size 2 \
  --output-json /workdir/npu_inference_profiling/lm_eval_score/outputs/single_runs/qwen36_35b_tp2.json
```

### 7. Determinism — two runs, same seed

Runs **two lightweight MMLU subjects** by default (override with `TASKS=...`):

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
/workdir/npu_inference_profiling/lm_eval_score/run_mmlu_full_twice_single_npu.sh
```

Outputs under `outputs/determinism/<timestamp>/` (`run_1.json`, `run_2.json`, `compare.txt`, `MANIFEST.txt`). Latest:

```bash
cat /workdir/npu_inference_profiling/lm_eval_score/outputs/determinism/LATEST_RUN_DIR.txt
```

Manual compare:

```bash
python3 /workdir/npu_inference_profiling/lm_eval_score/compare_mmlu_runs.py \
  /path/to/run_1.json /path/to/run_2.json
```

---

## Timing fields (`timing` in JSON; summary lines at top of `*_tables.txt`)

| Field | Meaning |
|-------|---------|
| `wall_clock_total_seconds` | Full `simple_evaluate` wall time |
| `vllm_llm_init_seconds` | Time inside `vllm.LLM.__init__` (engine / weights) |
| `lm_eval_model_setup_seconds` | Full lm-eval `VLLM` wrapper `__init__` |
| `tokenizer_config_overhead_seconds` | Setup minus `LLM` init |
| `eval_execution_seconds` | Harness + requests after model wrapper is ready |
| `eval_est_speed_input_toks_per_s_last` | Last **vLLM tqdm** running-average input rate (`Processed prompts`, same idea as log line `est. speed input: … toks/s`) |
| `eval_est_speed_output_toks_per_s_last` | Last output-token rate from that postfix |
| `eval_est_speed_input_toks_per_s_peak` | Peak input rate observed during eval |
| `eval_est_speed_output_toks_per_s_peak` | Peak output rate observed during eval |
| `eval_throughput_note` | Short explanation of how these relate to vLLM tqdm |

The `*_tables.txt` sidecar repeats throughput on line `vLLM_tqdm_est_speed_toks_per_s:` right after `timing_seconds:`.

---

## Note on `qwen3_5` / `qwen3_5_moe`

Dense checkpoints may use `model_type: qwen3_5` and MoE `qwen3_5_moe` before Transformers registers those strings. The runner maps local `config.json` `text_config` onto `Qwen3Config` / `Qwen3MoeConfig` so lm-eval can build the tokenizer; vLLM still loads the real weights.

## NPU selection

If **`ASCEND_RT_VISIBLE_DEVICES=0`** fails with insufficient free HBM, another process may own card `0`. Pick an idle index (check `npu-smi info`) e.g. `export ASCEND_RT_VISIBLE_DEVICES=1`.
