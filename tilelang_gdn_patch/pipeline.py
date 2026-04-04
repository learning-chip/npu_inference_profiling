"""
Compose TileLang ``opt_gdn`` kernels (vendored under ``tilelang_gdn_patch/opt_gdn/``).

Callers pass **Triton / PyTorch** layout ``[B, T, H, …]`` (``head_first=False``). Kernels expect
**head-first** ``[B, H, T, …]`` so that sequence slices are contiguous for Ascend ``T.copy`` (see
``NATIVE_BTH_LAYOUT.md``). This module performs a single ``transpose(1, 2).contiguous()`` on inputs
and the same on the output tensor so :mod:`tilelang_gdn_patch.api` stays transpose-free.
"""

from __future__ import annotations

import torch

from tilelang_gdn_patch.opt_gdn.opt_gdn_chunk_cumsum import cumsum_ker
from tilelang_gdn_patch.opt_gdn.opt_gdn_chunk_h import chunk_h_ker
from tilelang_gdn_patch.opt_gdn.opt_gdn_chunk_o import chunk_o_ker
from tilelang_gdn_patch.opt_gdn.opt_gdn_chunk_scaled_dot_kkt import kkt_ker
from tilelang_gdn_patch.opt_gdn.opt_gdn_solve_tril import solve_tril_64_ker
from tilelang_gdn_patch.opt_gdn.opt_gdn_wy_fast import wy_fast_ker


def _bth_to_bht(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def run_opt_gdn_tilelang_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q, k, v, g, beta: **``[B, T, H, …]``** (same as Triton); dtypes as produced by :mod:`api`.

    Returns:
        ``o`` shaped ``[B, T, H, DV]``, ``final_state`` ``[B, H, DK, DV]``.
    """
    q = _bth_to_bht(q)
    k = _bth_to_bht(k)
    v = _bth_to_bht(v)
    g = _bth_to_bht(g)
    beta = _bth_to_bht(beta)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    assert v.shape == (B, H, L, DV) and g.shape == (B, H, L) and beta.shape == (B, H, L)
    assert k.shape == (B, H, L, DK)

    C = chunk_size
    if L % 512 != 0:
        raise ValueError(f"TileLang cumsum requires L % 512 == 0; got L={L}.")
    if H % 2 != 0:
        raise ValueError(f"TileLang cumsum expects H % 2 == 0; got H={H}")

    BV = DV
    idt = torch.eye(C, device=q.device, dtype=torch.float32)
    msk1 = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.float32), diagonal=-1)
    msk2 = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.float32), diagonal=0)

    workspace = torch.zeros((B * H * ((DV + BV - 1) // BV), DK, BV), device=q.device, dtype=q.dtype)
    s = torch.zeros((B, H, (L + C - 1) // C, DK, DV), device=q.device, dtype=q.dtype)

    ker1 = cumsum_ker(B, H, L, C)
    ker2 = kkt_ker(B, H, L, DK, C)
    ker3 = solve_tril_64_ker(B, H, L)
    ker4 = wy_fast_ker(B, H, L, DK, DV, C)
    ker5 = chunk_h_ker(B, H, L, DK, DV, C)
    ker6 = chunk_o_ker(B, H, L, DK, DV, C)

    g_sum = ker1(g)
    a = ker2(k, beta, g_sum, msk1)
    a = ker3(a, idt)
    w, u = ker4(k, v, beta, g_sum, a)
    new_v, final_s = ker5(k, w, u, g_sum, workspace, s)
    o = ker6(q, k, new_v, s, g_sum, msk2)
    o = _bth_to_bht(o)
    return o, final_s
