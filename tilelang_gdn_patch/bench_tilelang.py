#!/usr/bin/env python3
"""Time the TileLang ``opt_gdn`` pipeline on NPU (median ms, proxy TFLOP/s and GiB/s)."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

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


def chunk_flops_proxy(t: int, h: int, dk: int, dv: int) -> float:
    return 32.0 * float(t * h * dk * dv)


def chunk_bytes_proxy(b: int, t: int, h: int, dk: int, dv: int) -> int:
    elem2, elem4 = 2, 4
    return (
        b * t * h * dk * elem2 * 2
        + b * t * h * dv * elem2
        + b * t * h * elem4
        + b * t * h * elem2
        + b * h * dk * dv * elem2
        + b * t * h * dv * elem2
    )


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
    args = p.parse_args()

    if args.t % 512 != 0:
        raise SystemExit("T must be divisible by 512.")

    import torch.nn.functional as F
    import torch_npu

    torch_npu.npu.set_device(args.device)

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tilelang_gdn_patch.pipeline import run_opt_gdn_tilelang_pipeline

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
    qh = qn.transpose(1, 2).contiguous().to(torch.float16)
    kh = kn.transpose(1, 2).contiguous().to(torch.float16)
    vh = v.transpose(1, 2).contiguous().to(torch.float16)
    gh = g_l.transpose(1, 2).contiguous()
    bh = beta.transpose(1, 2).contiguous().to(torch.float16)

    def run():
        return run_opt_gdn_tilelang_pipeline(qh, kh, vh, gh, bh, chunk_size=64)

    ms = _median_ms(run, args.warmup, args.repeats)
    total_t = B * T
    flops = chunk_flops_proxy(total_t, H, DK, DV)
    bytes_b = chunk_bytes_proxy(B, T, H, DK, DV)
    s = ms / 1000.0
    print(
        f"TileLang GDN chunk pipeline | B={B} T={T} H={H} DK={DK} DV={DV} | "
        f"median {ms:.3f} ms | ~{flops / s / 1e12:.2f} TFLOP/s (proxy) | "
        f"~{bytes_b / s / (1024**3):.2f} GiB/s (proxy)",
        flush=True,
    )


if __name__ == "__main__":
    main()
