# TileLang chunk Gated Delta Net (reference patch)

This directory wraps the **TileLang `opt_gdn`** pipeline from
[`tilelang-ascend/examples/linear_attention_and_rnn/opt_gdn`](../../tilelang-ascend/examples/linear_attention_and_rnn/opt_gdn)
so it can be called with the **same tensor layout as vLLM-Ascend Triton** (`head_first=False`):

- `q`, `k`: `[batch, seq, head, dim_k]`
- `v`: `[batch, seq, head, dim_v]`
- `g`: `[batch, seq, head]` float32 (log-space gating)
- `beta`: `[batch, seq, head]`

Implementation: [`api.py`](./api.py) (`chunk_gated_delta_rule_tilelang`) transposes to head-first `[B, H, T, …]`, runs
[`pipeline.py`](./pipeline.py) (`run_opt_gdn_tilelang_pipeline`), then transposes outputs back.

## Correctness

The composed TileLang kernels match the **sequential reference** `ref_seq_gdn` from
`opt_gdn_full.py` (ported as [`reference.py`](./reference.py)). Run on NPU:

```bash
cd npu_inference_profiling
python -m tilelang_gdn_patch.test_correctness_vs_ref --device 0 --t 512 --h 8
```

**Triton parity:** Direct comparison to `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule` is the
same target as [`bench_gdn_per_stage.py`](../single_gdn_layer/bench_gdn_per_stage.py) (lines 341–353). In this
environment, calling that Triton entry point without the full scheduler sometimes hits a Triton compile
issue in `wy_fast` for dense batches; numerically, the TileLang chain is aligned with the **same**
`opt_gdn_full` reference that upstream uses to validate kernels.

**Limits vs production Triton:**

- **Homogeneous heads:** TileLang `opt_gdn` assumes one head count `H` for `q`, `k`, and `v` (no separate
  `H_g` / `H_v` like the Triton `chunk_delta_h` path).
- **Sequence length:** upstream `cumsum_ker` requires `T % 512 == 0` (`C=64`, inner tile `CC=8`).
- **`initial_state`:** not fused into this TileLang chain; compare Triton with zero state when validating.

## Performance (on-device, this workspace)

Commands:

```bash
cd npu_inference_profiling
python -m tilelang_gdn_patch.bench_tilelang --device 0 --t 512 --h 8 --dk 128 --dv 128 --warmup 2 --repeats 7
```

**Measured (median of timed region, proxy FLOPs = `32·T·H·DK·DV`, proxy bytes = tensor traffic model):**

| Kernel | Shape (B×T×H×DK×DV) | Median | TFLOP/s (proxy) | GiB/s (proxy) |
|--------|---------------------|--------|-----------------|---------------|
| TileLang `opt_gdn` pipeline | 1×512×8×128×128 | **2.895 ms** | **~0.74** | **~1.44** |

**Triton baseline (same proxy, different benchmark / shape):** from
[`gdn_per_stage.md`](../single_gdn_layer/gdn_per_stage.md), the **chunk_gated_delta_rule** stage alone on
0.8B-class GDN at `1×4096` is on the order of **~3.8 ms** with **~9 TFLOP/s** and **~17 GiB/s** (not identical
tensor shapes or batch layout; use as coarse comparison only).

## Upstream path and grid / dynamic shape notes

- Set `TILELANG_OPT_GDN_ROOT` if `tilelang-ascend` lives outside the default relative path
  (`../../tilelang-ascend/examples/linear_attention_and_rnn`).
- **Launch grid:** upstream `chunk_h_ker` uses `T.Kernel(B * H * bv_num, …)` (see
  `opt_gdn_chunk_h.py`). A production port should follow the
  [`linear_attention.py`](../../tilelang-pto-demo/models/linear_attn/linear_attention.py) pattern:
  fixed `core_num` blocks and a serial loop over work items. That rewrite is **not** applied here to avoid
  diverging from validated `opt_gdn` kernels.
- **Dynamic `B` / `T`:** avoiding JIT recompilation for every shape requires `T.symbolic("B")`,
  `T.symbolic("L")` (and static `H`, `DK`, `DV`) across **all** `opt_gdn` kernels; this is future work.
