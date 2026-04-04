# GatedDeltaNet (Qwen3.5) per-stage cost — prefill

This file summarizes **on-device** timings from `bench_gdn_per_stage.py`, using the same setup as `bench_gdn_forward_random.py` (random weights, `GDNAttentionMetadataBuilder`, `set_forward_context`, patched `AscendQwen3_5GatedDeltaNet` / Ascend ops).

**Measured environment (this run):** Ascend NPU `npu:0`, `vllm-ascend` stack with Triton FLA kernels, warmup `3`, repeats `10`, median latency.

## What each stage is

| Stage | Implementation |
|--------|----------------|
| **input_proj** | `in_proj_qkvz` → split `mixed_qkv` / `z` reshape → `in_proj_ba` → `b`,`a` contiguous (matches `patch_qwen3_5.py` Part 1). |
| **causal_conv** | `torch.ops._C_ascend.npu_causal_conv1d_custom` on prefill (`num_prefills > 0`), same arguments as `_forward_core`. |
| **fused_gdn_gating** | `fused_gdn_gating_patch` (Triton) on `A_log`, `a`, `b`, `dt_bias`. |
| **chunk_gdr** | `chunk_gated_delta_rule` (patched `vllm_ascend.ops.triton.fla.chunk`) with `use_qk_l2norm_in_kernel=True`, `non_spec_chunked_prefill_meta`, `cu_seqlens=non_spec_query_start_loc`. |
| **output** | Gated `norm` → `rearrange` → `out_proj` (Part 3). Micro-bench uses **random** activations with the **same shapes** as the real path. |

**Decoding / speculative paths are not used** (prefill-only metadata).

## Important interpretation

- Each stage is timed **in isolation** (`torch.npu.synchronize` + `perf_counter`). Standalone times **do not add up** to the full layer: the real `gdn_attention_core` path overlaps memory traffic and scheduling, and micro-benches repeat kernels with fresh clones where needed (e.g. conv uses `mixed_qkv.clone()` each call).
- Percent columns are **(stage_ms / full_forward_ms) × 100**. They are **not** intended to sum to 100%.
- **fake vs real chunk I/O**: Random tensors with **identical shapes/dtypes** vs tensors from `conv → rearrange → gating`; medians matched within noise (validates shape/metadata correctness).

## Chunk kernel metrics (definitions)

- **TFLOP/s**: Uses the same **proxy** as `bench_gdn_forward.py` / README: **FLOPs = 32 × T × N<sub>v</sub> × D<sub>k</sub> × D<sub>v</sub>** (T = total tokens, N<sub>v</sub> value heads / TP, D<sub>k</sub>/D<sub>v</sub> head dims). **TFLOP/s = FLOPs / t_chunk / 1e12**.
- **GiB/s**: **Tensor traffic model** = sum of element sizes for `q`, `k`, `v`, `g`, `beta`, `initial_state`, and output `o` (same shapes as the real chunk call; `g` is float32, others mostly bf16). **GiB/s = bytes / t_chunk / 1024³**.

These are **roofline-style indicators**, not hardware counters.

---

## 0.8B-class config (hidden 1024, …)

| batch×seq (T) | full (ms) | in (ms) | conv (ms) | gate (ms) | chunk (ms) | out (ms) | %in | %conv | %gate | %chunk | %out | chunk TFLOP/s | chunk GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×4096 (T=4096) | 7.592 | 0.491 | 1.950 | 0.257 | 3.801 | 2.221 | 6.5% | 25.7% | 3.4% | 50.1% | 29.3% | 9.04 | 16.80 |
| 4×4096 (T=16384) | 14.625 | 1.070 | 2.687 | 0.264 | 7.901 | 8.685 | 7.3% | 18.4% | 1.8% | 54.0% | 59.4% | 17.40 | 32.32 |
| 8×2048 (T=16384) | 14.246 | 1.073 | 2.238 | 0.261 | 8.064 | 8.703 | 7.5% | 15.7% | 1.8% | 56.6% | 61.1% | 17.04 | 32.15 |

**Coarse buckets (same run, 1×4096):** relative to full forward ~7.59 ms — **conv ~26%**, **chunk ~50%**, **gating ~3%**; **input+output** micro-benches ~6.5% + ~29% (see non-additivity note above).

---

## 9B-class config (hidden 4096, …)

| batch×seq (T) | full (ms) | in (ms) | conv (ms) | gate (ms) | chunk (ms) | out (ms) | %in | %conv | %gate | %chunk | %out | chunk TFLOP/s | chunk GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×4096 (T=4096) | 10.855 | 1.538 | 2.638 | 0.260 | 5.414 | 4.567 | 14.2% | 24.3% | 2.4% | 49.9% | 42.1% | 12.69 | 17.81 |

---

## How to reproduce

```bash
cd npu_inference_profiling/single_gdn_layer
python bench_gdn_per_stage.py --variant 0.8B --shapes 1x4096 4x4096 8x2048 --markdown
python bench_gdn_per_stage.py --variant 9B --shapes 1x4096 --markdown
```
