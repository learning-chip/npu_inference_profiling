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
| `compare_prefill_next_token.py` | Greedy tokens + first-step full-vocab logprobs: **`record`** / **`compare`** (**`--quantization ascend`** optional) |
| `benchmark_prefill_latency.py` | Prefill TTFT sweep → JSON lines (**`--quantization ascend`** optional) |
| `run_benchmark_prefill_three_way.sh` | Writes `OUT_DIR/<LABEL>/{triton,pto,pto_mega}.jsonl`; env **`BENCH_QUANTIZATION`** forwards to **`benchmark_prefill_latency.py`** |
| `run_compare_prefill.sh` | Two shell-level `python … record` invocations + `compare` (no nested Python `subprocess`) |
| `compare_pto_triton_lm_eval.py` | Wikitext token PPL + optional MMLU (file-based child results; see script doc) |
| `sanity_pto_triton_eval.sh` | Tiny Bash-driven eval with `timeout(1)` |
| `vllm_source_patch/` | How to patch installed `vllm_ascend` worker imports |

## Prefill parity (`compare_prefill_next_token.py`)

vLLM already spawns engine subprocesses. This repo **does not** nest another Python `subprocess.run` around each backend (avoids `Popen.communicate` stalls when worker logs fill pipes).

**Usage**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export BENCH_QUANTIZATION=ascend   # required for msmodelslim W8A8 paths (e.g. **Qwen3.6-27B-w8a8**) in default ``BENCHMARK_MODEL_SPECS``
./run_compare_prefill.sh
```

Default **`BENCHMARK_MODEL_SPECS`** lists **2B**, **0.8B**, **4B**, **9B**, and **27B-w8a8** (see **`run_compare_prefill.sh`** / **`run_benchmark_prefill_three_way.sh`**). **Qwen3.6-35B-A3B-w8a8** MoE is **not** in that default list (heavy load); pass an explicit **`BENCHMARK_MODEL_SPECS`** when benchmarking it. Set **`BENCH_QUANTIZATION=ascend`** whenever a W8A8 msmodelslim path is included; override **`BENCHMARK_MODEL_SPECS`** if you only run BF16 checkpoints and do not want **`ascend`** quantization.

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
**Run:** `bench_prefill_20260428_gqa` — same **sequence-length ladder** and **`WARMUP` / `REPEATS`** as **`bench_prefill_20260424`** (0.8B / 2B): **`SEQ_LENS="512 1024 2048 4096 8192 16384 32768 65536"`**, `WARMUP=2`, `REPEATS=10`, `ASCEND_RT_VISIBLE_DEVICES=0`, **`run_benchmark_prefill_three_way.sh`** with **`BENCHMARK_MODEL_SPECS`** for 4B/9B only — `bench_prefill_20260428_gqa/<LABEL>/{triton,pto,pto_mega}.jsonl` plus `run.log`.

**PTO megakernel** routinely beats Ascend **Triton** on median TTFT from mid-length prompts upward; staged **PTO** is fastest on several long-context rows for **4B** here. Figures: **`figure/prefill_speedup_4B.png`** and **`prefill_speedup_9B.png`** from **`python3 plot_speedup.py`**.

### Qwen3.5-4B

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|--------------------:|--------------------------:|
| 512 | 182.2 | 187.4 | 153.0 |
| 1024 | 189.0 | 190.1 | 154.5 |
| 2048 | 220.3 | 219.1 | 180.0 |
| 4096 | 319.6 | 303.1 | 279.1 |
| 8192 | 561.6 | 514.0 | 500.1 |
| 16384 | 1081.3 | 975.3 | 958.3 |
| 32768 | 2261.5 | 2061.9 | 2043.8 |
| 65536 | 5527.5 | 5129.9 | 5100.1 |

### Qwen3.5-9B

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|--------------------:|--------------------------:|
| 512 | 177.1 | 186.2 | 149.4 |
| 1024 | 198.7 | 203.5 | 170.0 |
| 2048 | 258.2 | 256.5 | 222.6 |
| 4096 | 410.9 | 402.7 | 371.7 |
| 8192 | 732.1 | 688.8 | 663.8 |
| 16384 | 1431.9 | 1340.1 | 1326.2 |
| 32768 | 3060.1 | 2878.6 | 2862.9 |
| 65536 | 7159.3 | 6776.0 | 6734.7 |

### Qwen3.6-27B W8A8 (`quantization=ascend`, msmodelslim)

Checkpoint: **`/scratch/model_weights/Qwen3.6-27B-w8a8`**. **`text_config`** matches Qwen3.5 dense GQA (**Hg=16**, **48** recurrent value heads, **D=128** KV → **dynamic_bsnd_groupvalue** / **`pto_mega_kernel_groupvalue`**; megakernel build logs **`H48_Hg16_D128`**).

Use **`--quantization ascend`** on **`benchmark_prefill_latency.py`** and **`compare_prefill_next_token.py record`**, or **`BENCH_QUANTIZATION=ascend`** when running **`run_benchmark_prefill_three_way.sh`** (forwards **`--quantization`** to **`LLM()`**).

| Parity probe (`SEQ_LEN=512`, `NUM_GEN=5`) | Result |
|------------------------------------------|--------|
| Greedy continuation (5 decoded tokens) | **Match** Ascend **Triton** vs **PTO staged** vs **PTO mega** (`[279, 15217, 5388, 13, 561]`). |
| First-step full-vocab logprob vs **Triton** (`--max-logprobs 300000`) | `max_abs ≈ 6.99`, `rmse ≈ 1.31` — typical **`numpy.allclose`** default strict check **does not apply** vs BF16-only models; quantized matmul/stacking widens deltas while **the greedy path stays identical**. |
| **PTO staged** vs **PTO mega** first-step logits | **Bit-identical** on this checkpoint (`max_abs` difference 0). |

Benchmark JSONL (**`WARMUP=2`**, **`REPEATS=10`**): **`bench_prefill_Qwen36_27B_w8a8/27B-w8a8/`**. Ascend **Triton** **`chunk_gated_delta_rule`** failed at **`SEQ_LEN=65536`** (**`ACL … 507014`**) on this workload, so **`triton.jsonl`** stops at **32768**; **`pto.jsonl`** / **`pto_mega.jsonl`** include an extra row at **65536**. **`figure/prefill_speedup_27B-w8a8.png`** (**`plot_speedup.py`**) plots **median TTFT** and **input TPS** for **PTO** / **PTO mega** through **65536**, shades the segment after the last Triton measurement, annotates it, and keeps **speedup vs Triton** only where the baseline exists (**`seq_len` ≤ longest Triton run**).

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|---------------------:|--------------------------:|
| 512 | 354.9 | 365.0 | 306.9 |
| 1024 | 414.1 | 420.6 | 355.8 |
| 2048 | 557.1 | 546.1 | 487.2 |
| 4096 | 924.9 | 851.0 | 814.7 |
| 8192 | 1789.3 | 1653.8 | 1619.9 |
| 16384 | 3389.6 | 3134.8 | 3095.8 |
| 32768 | 7372.7 | 6890.2 | 6844.5 |
| 65536 | — (baseline failed) | 16386.5 | 16269.9 |

**Figure:** **`figure/prefill_speedup_27B-w8a8.png`** (PTO timings at **65536** plotted; shaded region + plot subtitle explain missing Triton).

### Qwen3.6-35B-A3B W8A8 MoE (`quantization=ascend`, msmodelslim)

Checkpoint **`/scratch/model_weights/Qwen3.6-35B-A3B-w8a8/`** (`Qwen3_5MoeForConditionalGeneration`). Recurrent linear layers use **GQA** groupvalue PTO (**Hg=16**, **32** value heads, **D=128** KV in `text_config`), same style as **4B/9B/27B** but with MoE FFNs.

Run with an explicit model spec, e.g. **`BENCHMARK_MODEL_SPECS="35B-A3B-w8a8:/scratch/model_weights/Qwen3.6-35B-A3B-w8a8/"`** and **`BENCH_QUANTIZATION=ascend`** (not part of the default multi-model driver list).

| Parity probe (`SEQ_LEN=512`, `NUM_GEN=5`) | Result |
|------------------------------------------|--------|
| Greedy continuation (5 decoded tokens) | **Match** Ascend **Triton** vs **PTO staged** vs **PTO mega** (`[279, 15217, 5388, 13, 561]`). |
| First-step full-vocab logprob vs **Triton** | **PTO staged:** `max_abs ≈ 1.65`, `rmse ≈ 0.47`. **PTO mega:** `max_abs ≈ 1.22`, `rmse ≈ 0.34` — W8A8 stack; strict **`numpy.allclose`** vs Triton **not** expected. |
| **PTO staged** vs **PTO mega** first-step logits | **Not** bit-identical here (`max_abs ≈ 1.28`, `rmse ≈ 0.26`) — differs from dense **27B**; MoE + two kernel paths can diverge in logits while **greedy** IDs still agree. |

Benchmark JSONL (**`WARMUP=2`**, **`REPEATS=10`**): **`bench_prefill_Qwen36_35B_A3B_w8a8/35B-A3B-w8a8/`**. On this run, **Triton** completed the same **seq_len** ladder through **65536** (no missing baseline row).

| seq_len | Triton median TTFT (ms) | PTO median TTFT (ms) | PTO mega median TTFT (ms) |
|--------:|-------------------------:|---------------------:|--------------------------:|
| 512 | 301.5 | 346.1 | 262.6 |
| 1024 | 305.0 | 341.3 | 263.4 |
| 2048 | 323.8 | 348.0 | 276.1 |
| 4096 | 388.6 | 382.3 | 317.4 |
| 8192 | 620.2 | 567.9 | 534.5 |
| 16384 | 1171.6 | 1041.8 | 1013.9 |
| 32768 | 2454.5 | 2196.7 | 2157.2 |
| 65536 | 5787.1 | 5298.3 | 5236.2 |

**Figure:** **`figure/prefill_speedup_35B-A3B-w8a8.png`**.

Parity artifacts (local run): **`_parity_q36_35b_a3b_w8a8/35B-A3B-w8a8/`** (`compare.txt`, **`compare_mega.txt`**, `*.npz`).

## Worker hook

Set `VLLM_PTO_PATCH_DIR` to this directory before starting vLLM so `vllm_ascend` can import `apply_pto_patch()`. See `vllm_source_patch/README.md` for patching an installed `vllm_ascend` tree (worker `__init__.py` + Qwen worker modules).
