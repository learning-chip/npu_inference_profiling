# PTO patch for vLLM-Ascend (Qwen3.5 GDN)

Out-of-tree replacement for FLA **chunk_gated_delta_rule** prefill with PTO JIT kernels, plus small validation scripts.

## GQA vs MHA (**Qwen3.5 sizes**)

Ascend lays out recurrent GDN tensors as **`[B, T, H, D]`** where **fewer heads** apply to projected **Q/K** (**Hg**) than to **values** and gated factors (**H**) (`text_config.linear_num_key_heads`, `linear_num_value_heads`; vLLM’s `AscendQwen3_5GatedDeltaNet.rearrange_mixed_qkv`).  
When **`H_kv < H`** (e.g. Qwen3.5-4B/9B: **Hg = 16**, **H = 32**), **`pto_chunk_gated_delta_rule.py`** routes to **`dynamic_bsnd_groupvalue`** staged kernels / **`pto_mega_kernel_groupvalue`** (same decomposition as **`verify_pto_triton_e2e_groupvalue`**).  
Legacy **`dynamic_bsnd`** / **`pto_mega_kernel`** remain used when **`H_kv == H_v`** (**0.8B**, **2B**).

Different **per-head widths** (**`linear_key_head_dim`** ≠ **`linear_value_head_dim`**) automatically fall through to Ascend **Triton** (PTO JIT templates compile a single head size **D**).

## Driver **`apply_pto_patch`**

Scripts that **`import vllm`** in the coordinator process **must call** `adapt_patch(False)` first, **`apply_pto_patch()`** second (**after `fla.ops` is finalized** — see **`compare_prefill_next_token._verify_chunk_backend`**). Workers still use `VLLM_PTO_PATCH_DIR` from **`vllm_ascend/patch/worker/__init__.py`** per **`vllm_source_patch/`**.

### Layout

| Path | Role |
|------|------|
| `apply.py` | `apply_pto_patch()` — bind PTO wrapper on `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` |
| `pto_chunk_gated_delta_rule.py` | PTO forward (**MHA** ↔ **dynamic_bsnd**/megakernel **GQA** ↔ **dynamic_bsnd_groupvalue**/megakernel gv), `record_function("PTO_gdn_*")` scopes |
| `compare_prefill_next_token.py` | Greedy tokens + first-step full-vocab logprobs: **`record`** / **`compare`** |
| `run_compare_prefill.sh` | Two shell-level `python … record` invocations + `compare` (no nested Python `subprocess`) |
| `compare_pto_triton_lm_eval.py` | Wikitext token PPL + optional MMLU (file-based child results; see script doc) |
| `sanity_pto_triton_eval.sh` | Tiny Bash-driven eval with `timeout(1)` |
| `vllm_source_patch/` | How to patch installed `vllm_ascend` worker imports |

## Prefill parity (`compare_prefill_next_token.py`)

vLLM already spawns engine subprocesses. This repo **does not** nest another Python `subprocess.run` around each backend (avoids `Popen.communicate` stalls when worker logs fill pipes).

**Usage**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
./run_compare_prefill.sh
```

Or manually:

```bash
python3 compare_prefill_next_token.py record --backend triton --output /tmp/tri.npz --device 0
python3 compare_prefill_next_token.py record --backend pto   --output /tmp/pto.npz --device 0
python3 compare_prefill_next_token.py compare /tmp/tri.npz /tmp/pto.npz
```

### Last measured parity (2026-04-24, Ascend 910B, NPU 0)

Command: `OUTDIR=/tmp/prefill_cmp_test3 SEQ_LEN=128 NUM_GEN=11 ./run_compare_prefill.sh`  
Model: `Qwen3.5-0.8B` snapshot `2fc06364715b967f1860aea9cf38778875588b17`  
`max_logprobs=300000`, `SamplingParams(logprobs=-1)` on first decode step.

| Check | Result |
|--------|--------|
| Greedy token IDs (11 tokens) | **Identical** Triton vs PTO: `[15217, 5388, 13, 561, 3841, 13477, 37550, 33075, 888, 279, 15217]` |
| First-step logprob vector | `max_abs = 0.218746`, `rmse = 0.0433051`, `248320 / 248320` finite entries |
| `numpy.allclose` | **PASS** with default `atol=0.005`, `rtol=0.02` |

### GQA spot-check — **Qwen3.5-4B** (Ascend NPU 0)

**Worker must use the patched `vllm_ascend` imports** (`vllm_source_patch/apply_vllm_ascend_pto_hook.py` applied to **`patch/worker/__init__.py`** and **`patch_qwen3_{5,_next}.py`**) so engine workers bind **`vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`**, not a stale **`fla.ops`** entry. Earlier runs showing **`max_abs=0`** versus Triton and **tiny TTFT differences** matched **silent fallback**: e.g. `pto_dynamic_common` threw **`AssertionError: Invalid device id`** during dynamic-kernel import in the worker (**only `RuntimeError` was caught**) so the coordinator could patch the driver process while **`EngineCore`** still failed once PTO executed—or both paths matched Triton. That exception class is handled in-repo now (fallback `BLOCK_DIM=24`).

After the hook plus that import-time guard:

Command:  
`SEQ_LEN=32 NUM_GEN=3` (`max_logprobs=300000`)  
`compare_prefill_next_token.py record … --backend {triton,pto}` then `compare`:

| Bench | Greedy tokens | First-step logprobs (full vocab) |
|-------|---------------|--------------------------------|
| Triton vs PTO | **Same** `[561, 3841, 13477]` | `max_abs≈0.150`, `rmse≈0.035` (finite; **not** bit-identical) |

`numpy.allclose` at `atol=0.005`, `rtol=0.02` **fails** on the first-step logprob vector—expected for PTO vs Triton.

## Prefill TTFT benchmark (GQA **4B** / **9B**)

**Metric:** `RequestOutput.metrics.first_token_latency` (seconds), reported as **median TTFT** (ms) in JSONL (`median_ttft_ms`) and implied **input_tps** = `seq_len / (mean_ttft_ms / 1000)` from `benchmark_prefill_latency.py`.  
**Run:** 2026-04-28 (fresh rerun), `WARMUP=1`, `REPEATS=5`, `SEQ_LENS="512 2048 8192 32768"`, `ASCEND_RT_VISIBLE_DEVICES=0`, **`run_benchmark_prefill_three_way.sh`** with `BENCHMARK_MODEL_SPECS` restricted to 4B/9B, output `bench_prefill_20260428_gqa/<LABEL>/{triton,pto,pto_mega}.jsonl` plus `run.log`.

At short sequence lengths, Triton vs staged PTO vs mega can sit within a few milliseconds; at **8192** and **32768** tokens, **PTO megakernel** shows clearly **lower median TTFT** than Ascend Triton (staged PTO is also faster than Triton at 4B on long prompts in this run). Regenerate **`figure/prefill_speedup_4B.png`** / **`prefill_speedup_9B.png`** with **`python3 plot_speedup.py`**.

### Qwen3.5-4B

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|--------------------:|--------------------------:|
| 512 | 177.5 | 191.7 | 155.1 |
| 2048 | 217.7 | 219.3 | 181.5 |
| 8192 | 560.3 | 515.0 | 493.8 |
| 32768 | 2256.6 | 2046.3 | 2030.3 |

### Qwen3.5-9B

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|--------------------:|--------------------------:|
| 512 | 178.8 | 183.7 | 155.0 |
| 2048 | 257.0 | 256.0 | 222.1 |
| 8192 | 727.4 | 681.9 | 662.4 |
| 32768 | 3027.6 | 2843.9 | 2823.3 |

## Worker hook

Set `VLLM_PTO_PATCH_DIR` to this directory before starting vLLM so `vllm_ascend` can import `apply_pto_patch()`. See `vllm_source_patch/README.md` for patching an installed `vllm_ascend` tree (worker `__init__.py` + Qwen worker modules).
