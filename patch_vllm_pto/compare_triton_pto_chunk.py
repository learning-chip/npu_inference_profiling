#!/usr/bin/env python3
"""Compare Ascend Triton vs PTO ``chunk_gated_delta_rule`` on random GDN tensors (NPU)."""
from __future__ import annotations

import argparse
import os
import sys

_WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKDIR not in sys.path:
    sys.path.insert(0, _WORKDIR)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:4")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--D", type=int, default=128)
    args = p.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    _maybe_pto = os.path.join(os.path.dirname(__file__))
    if _maybe_pto not in sys.path:
        sys.path.insert(0, _maybe_pto)

    torch = __import__("torch")
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)

    import vllm_ascend  # noqa: F401 — worker patches
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()

    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule as triton_fn

    from pto_chunk_gated_delta_rule import bind_triton

    pto_fn = bind_triton(triton_fn)

    T, H, D = args.T, args.H, args.D
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    q = torch.randn(1, T, H, D, generator=gen, dtype=torch.float32).to(dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, generator=gen, dtype=torch.float32).to(dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, generator=gen, dtype=torch.float32).to(dev, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, H, generator=gen, dtype=torch.float32).to(dev, dtype=torch.bfloat16))
    g_log = torch.nn.functional.logsigmoid(
        torch.randn(1, T, H, generator=gen, dtype=torch.float32).to(dev)
    )
    cu = torch.tensor([0, T], dtype=torch.long, device=dev)
    z0 = torch.zeros(1, H, D, D, device=dev, dtype=torch.bfloat16)

    kwargs = dict(
        scale=D**-0.5,
        initial_state=z0,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    from types import SimpleNamespace
    from unittest import mock

    from vllm.forward_context import ForwardContext, override_forward_context

    _fc = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        virtual_engine=0,
    )
    _pcp = SimpleNamespace(world_size=1, rank_in_group=0)
    with override_forward_context(_fc), mock.patch(
        "vllm_ascend.ops.triton.fla.chunk.get_pcp_group", return_value=_pcp
    ):
        o_t, fs_t = triton_fn(q, k, v, g_log, beta, **kwargs)
        torch.npu.synchronize()
        o_p, fs_p = pto_fn(q, k, v, g_log, beta, **kwargs)
    torch.npu.synchronize()

    def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
        d = (a.float() - b.float()).abs()
        mx = float(d.max().item())
        rmse = float(torch.sqrt((d**2).mean()).item())
        print(f"{name}: max_abs={mx:.6g} rmse={rmse:.6g}")
        return mx

    mx_o = _report("o", o_t, o_p)
    mx_fs = _report("final_state", fs_t, fs_p) if fs_t is not None and fs_p is not None else 0.0
    ok = mx_o < 0.05 and mx_fs < 0.15
    if not ok:
        print("FAIL: thresholds o<0.05, fs<0.15 (bf16 vs fp16 PTO stack)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
