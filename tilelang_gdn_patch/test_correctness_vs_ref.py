#!/usr/bin/env python3
"""Assert TileLang pipeline matches ``reference.ref_seq_gdn_bth`` (sequential GDN in ``[B,T,H,…]``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--t", type=int, default=512)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.t % 512 != 0:
        raise SystemExit("T must be divisible by 512 (TileLang cumsum).")

    import torch.nn.functional as F
    import torch_npu

    torch_npu.npu.set_device(args.device)

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tilelang_gdn_patch.api import chunk_gated_delta_rule_tilelang
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

    o_tl, _fs = chunk_gated_delta_rule_tilelang(
        q,
        k,
        v,
        g_l,
        beta,
        initial_state=None,
        use_qk_l2norm_in_kernel=True,
        chunk_size=64,
    )

    qn = F.normalize(q.float(), dim=-1).to(q.dtype)
    kn = F.normalize(k.float(), dim=-1).to(k.dtype)
    ref = ref_seq_gdn_bth(qn, kn, v, g_l, beta)

    torch.testing.assert_close(
        o_tl.float().cpu(),
        ref.float().cpu(),
        rtol=2e-2,
        atol=2e-2,
    )
    print("OK: TileLang pipeline matches ref_seq_gdn_bth.")


if __name__ == "__main__":
    main()
