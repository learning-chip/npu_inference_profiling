#!/usr/bin/env python3
"""
TileLang ``opt_gdn`` vs Triton vs ``ref_seq_gdn_bth`` on shared tensors.

Triton uses :func:`tilelang_gdn_patch.triton_gdn_simple.chunk_gated_delta_rule_simple` (varlen ``cu_seqlens``),
a local wrapper around upstream Triton kernels without vLLM ``ForwardContext`` / PCP hooks.

Use **packed varlen** (``cu_seqlens``) because the dense path can fail to compile on Ascend; use
``scale=1.0`` to match this TileLang patch (default Triton scale is ``1/sqrt(DK)``).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable

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


def _proxy_flops(t: int, h: int, dk: int, dv: int) -> float:
    return 32.0 * float(t * h * dk * dv)


def _proxy_bytes(b: int, t: int, h: int, dk: int, dv: int) -> int:
    e2, e4 = 2, 4
    return (
        b * t * h * dk * e2 * 2
        + b * t * h * dv * e2
        + b * t * h * e4
        + b * t * h * e2
        + b * h * dk * dv * e2
        + b * t * h * dv * e2
    )


def _triton_packed_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Build vLLM varlen inputs (``batch == 1``) so Triton uses the working code path.

    Returns ``(q1, k1, v1, g1, b1, cu_seqlens, unpad_o)`` where ``unpad_o`` maps Triton
    output ``(1, B*T, H, DV)`` back to ``(B, T, H, DV)`` when ``B > 1``.
    """
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    if B == 1:
        cu = torch.tensor([0, T], dtype=torch.int32, device=device)

        def unpad_o(o: torch.Tensor) -> torch.Tensor:
            return o

        return q, k, v, g, beta, cu, unpad_o

    q1 = q.reshape(1, B * T, H, DK).contiguous()
    k1 = k.reshape(1, B * T, H, DK).contiguous()
    v1 = v.reshape(1, B * T, H, DV).contiguous()
    g1 = g.reshape(1, B * T, H).contiguous()
    b1 = beta.reshape(1, B * T, H).contiguous()
    cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)

    def unpad_o_bt(o: torch.Tensor) -> torch.Tensor:
        return o.reshape(B, T, H, DV)

    return q1, k1, v1, g1, b1, cu, unpad_o_bt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--t", type=int, default=512)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    args = p.parse_args()

    if args.t % 512 != 0:
        raise SystemExit("TileLang cumsum requires T % 512 == 0.")

    import torch.nn.functional as F
    import torch_npu

    torch_npu.npu.set_device(args.device)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    import vllm_ascend.patch.worker  # noqa: F401

    from tilelang_gdn_patch.api import chunk_gated_delta_rule_tilelang
    from tilelang_gdn_patch.pipeline import run_opt_gdn_tilelang_pipeline
    from tilelang_gdn_patch.reference import ref_seq_gdn_bth

    torch.manual_seed(args.seed)
    dev = torch.device("npu", args.device)
    B, T, H, DK, DV = args.b, args.t, args.h, args.dk, args.dv

    q = torch.randn(B, T, H, DK, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, DK, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, DV, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, dtype=torch.float32, device=dev)
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)

    g_l = F.logsigmoid(g)
    qn = F.normalize(q.float(), dim=-1).to(q.dtype)
    kn = F.normalize(k.float(), dim=-1).to(k.dtype)
    qh = qn.to(torch.float16)
    kh = kn.to(torch.float16)
    vh = v.to(torch.float16)
    gh = g_l.to(torch.float32)
    bh = beta.to(torch.float16)

    # vLLM/Triton and TileLang both expect g in log-space (see chunk_gated_delta_rule docstring).
    o_tl, fs_tl = chunk_gated_delta_rule_tilelang(
        q,
        k,
        v,
        g_l,
        beta,
        initial_state=None,
        use_qk_l2norm_in_kernel=True,
        chunk_size=64,
    )
    ref = ref_seq_gdn_bth(qn, kn, v, g_l, beta)

    torch.testing.assert_close(
        o_tl.float().cpu(),
        ref.float().cpu(),
        rtol=args.rtol,
        atol=args.atol,
    )
    print("OK: TileLang vs ref_seq_gdn_bth (same tensors as Triton path would use).", flush=True)

    h0 = torch.zeros(B, H, DK, DV, dtype=torch.bfloat16, device=dev)
    triton_ok = False
    try:
        from tilelang_gdn_patch.triton_gdn_simple import chunk_gated_delta_rule_simple

        q_t, k_t, v_t, g_t, b_t, cu_seqlens, unpad_o = _triton_packed_inputs(
            q, k, v, g_l, beta, device=dev
        )

        def run_triton():
            o, fs = chunk_gated_delta_rule_simple(
                q=q_t,
                k=k_t,
                v=v_t,
                g=g_t,
                beta=b_t,
                scale=1.0,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            return unpad_o(o), fs

        o_tr, fs_tr = run_triton()
        torch.testing.assert_close(
            o_tr.float().cpu(),
            o_tl.float().cpu(),
            rtol=args.rtol,
            atol=args.atol,
        )
        torch.testing.assert_close(
            fs_tr.float().cpu(),
            fs_tl.float().cpu(),
            rtol=args.rtol,
            atol=args.atol,
        )
        print("OK: Triton vs TileLang.", flush=True)
        triton_ok = True
        ms_tr = _median_ms(lambda: run_triton()[0], args.warmup, args.repeats)
    except Exception as e:
        print(f"[Triton] skipped or failed: {type(e).__name__}: {e}", flush=True)

    ms_tl = _median_ms(
        lambda: run_opt_gdn_tilelang_pipeline(qh, kh, vh, gh, bh, chunk_size=64)[0],
        args.warmup,
        args.repeats,
    )
    tot = B * T
    f = _proxy_flops(tot, H, DK, DV)
    by = _proxy_bytes(B, T, H, DK, DV)
    s = ms_tl / 1000.0
    print(
        f"TileLang   {ms_tl:8.3f} ms | ~{f / s / 1e12:6.2f} TFLOP/s (proxy) | "
        f"~{by / s / (1024**3):6.2f} GiB/s (proxy)",
        flush=True,
    )
    if triton_ok:
        s2 = ms_tr / 1000.0
        print(
            f"Triton     {ms_tr:8.3f} ms | ~{f / s2 / 1e12:6.2f} TFLOP/s (proxy) | "
            f"~{by / s2 / (1024**3):6.2f} GiB/s (proxy)",
            flush=True,
        )


if __name__ == "__main__":
    main()
