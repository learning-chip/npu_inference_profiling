# PTO kernels in vLLM-Ascend Qwen3.5 prefill

## Goal

Replace the six Triton FLA kernels in `chunk_gated_delta_rule_fwd` (see `vllm_ascend/ops/triton/fla/chunk.py`) with the optimized PTO JIT chain from `pto-kernels/examples/jit_cpp/chunk_gdn/dynamic_bsnd` plus `fast_inverse` for `(I+L)^{-1}`.

## Design

- **Wrapper**: `pto_chunk_gated_delta_rule.py` exposes the same API as `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule`, delegating to PTO when safe and to Triton otherwise.
- **Monkey-patch target**: `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` (already redirected to Ascend Triton by `patch_triton.py`).
- **Worker import hook**: `VLLM_PTO_PATCH_DIR` must point at this directory. Installed `vllm_ascend/patch/worker/__init__.py` appends a small block that `import apply; apply_pto_patch()` after the stock Triton patch so **spawned engine workers** load the PTO wrapper. The same hook exists in the workspace copy under `/workdir/vllm-ascend/…` for source installs.
- **Chunk size**: PTO chain uses **C=128**; Triton uses C=64. Numerical agreement is validated against Triton on the same tensors (see below).

## Fallbacks (Triton path)

- `get_pcp_group().world_size > 1` (prefill context parallel).
- Non-NPU device.
- Non-zero `initial_state` (PTO `chunk_h` has no h0 hook yet).
- Missing `cu_seqlens` (PTO kernels are varlen-oriented like the Qwen3.5 prefill call site).

## Profiler labels

`record_function` scopes: `PTO_gdn_chunk_cumsum`, `PTO_gdn_scaled_dot_kkt`, `PTO_gdn_solve_tril`, `PTO_gdn_wy_fast`, `PTO_gdn_chunk_h`, `PTO_gdn_chunk_o`. Ascend Chrome traces are device-oriented; use engine logs and JIT `.so` artifacts under `dynamic_bsnd/` / `fast_inverse/` to confirm execution.

## Validation runs (2026-04-24)

| Check | Command / note | Result |
|--------|------------------|--------|
| Op-level Triton vs PTO | `python3 compare_triton_pto_chunk.py --device npu:4 --T 256` (+ forward context + PCP mock in-script) | `max_abs` on `o` ~5e-4, `final_state` RMSE small — **PASS** |
| Greedy decode + first-step logprobs E2E | `python3 compare_prefill_next_token.py --device 4 --seq-len 128 --num-generated 11` | 11 token IDs identical; full-vocab first-step logprobs `max_abs=0` — **PASS** |
| Profile + patch | `ASCEND_RT_VISIBLE_DEVICES=4 VLLM_PTO_PATCH_DIR=… python3 …/profile_qwen35_prefill.py …` | Engine log: “PTO chunk_gated_delta_rule patch is active”; trace written under `--profile-dir` |

## Usage

```bash
export ASCEND_RT_VISIBLE_DEVICES=4   # pick a free NPU
export VLLM_PTO_PATCH_DIR=/workdir/npu_inference_profiling/patch_vllm_pto
# optional: export PTO_LIB_PATH=… if headers live outside CANN defaults

python3 npu_inference_profiling/qwen35_prefill/profile_qwen35_prefill.py --pto --batch-size 1 --seq-len 4096
```

`--pto` sets `VLLM_PTO_PATCH_DIR` before workers spawn (see early `sys.argv` scan in the profile script).

## Files

| File | Role |
|------|------|
| `apply.py` | `apply_pto_patch()` replaces `fla.ops.chunk_gated_delta_rule` with `bind_triton(…)`. |
| `pto_chunk_gated_delta_rule.py` | PTO forward + fallbacks + profiler scopes. |
| `compare_triton_pto_chunk.py` | Direct op comparison (needs forward context stub). |
| `compare_prefill_next_token.py` | Two-subprocess greedy-token parity on Qwen3.5-0.8B. |

## Known gaps

- **Initial recurrent state**: non-zero `initial_state` still uses Triton; extend PTO `chunk_h` if full parity with continued prefill is required.
- **Unknown vLLM env**: `VLLM_PTO_PATCH_DIR` triggers vLLM “unknown environment variable” info log; harmless.
