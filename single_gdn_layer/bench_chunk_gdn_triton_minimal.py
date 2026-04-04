#!/usr/bin/env python3
"""
Minimal ``chunk_gated_delta_rule_simple`` timing on NPU (fake data), same call pattern as
``vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_gated_delta_rule.py``.
Uses :mod:`tilelang_gdn_patch.triton_gdn_simple` (self-contained single-device wrapper).

Requires ``PYTHONPATH`` so ``tilelang_gdn_patch`` and ``vllm-ascend`` are importable (kernels live in vllm-ascend).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    xs: list[float] = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(xs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()

    import torch_npu

    torch_npu.npu.set_device(args.device)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    import vllm_ascend.patch.worker  # noqa: F401

    from tilelang_gdn_patch.triton_gdn_simple import chunk_gated_delta_rule_simple

    dev = torch.device("npu", args.device)
    # Same shape as test_chunk_gated_delta_rule (varlen API exercised separately there).
    q = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, 17, 8, 128, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, 17, 8, dtype=torch.float32, device=dev)
    g = torch.nn.functional.logsigmoid(g)
    beta = torch.randn(1, 17, 8, dtype=torch.bfloat16, device=dev)
    h0 = torch.randn(3, 8, 128, 128, dtype=torch.bfloat16, device=dev)
    q_start_loc = torch.arange(0, 4, dtype=torch.int32, device=dev)

    def run_once():
        return chunk_gated_delta_rule_simple(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=1.0,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=q_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    try:
        o, fs = run_once()
    except Exception as e:
        print(f"chunk_gated_delta_rule_simple failed: {type(e).__name__}: {e}")
        raise
    print(f"out {tuple(o.shape)}, final_state {tuple(fs.shape)}")

    ms = _median_ms(lambda: run_once()[0], args.warmup, args.repeats)
    print(f"median forward {ms:.3f} ms (output only timed)")


if __name__ == "__main__":
    main()
