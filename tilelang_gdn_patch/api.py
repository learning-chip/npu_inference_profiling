"""
Public API: Triton layout ``[batch, seq, head, dim]`` (``head_first=False``).

Inputs are **``[B, T, H, …]``** like Triton. :mod:`tilelang_gdn_patch.pipeline` converts to kernel
layout (see ``NATIVE_BTH_LAYOUT.md``); this module only adjusts dtypes.
Optional **varlen** (``cu_seqlens``, ``B=1``) runs one TileLang pipeline per segment and concatenates.
"""

from __future__ import annotations

import torch

from tilelang_gdn_patch.pipeline import run_opt_gdn_tilelang_pipeline


def chunk_gated_delta_rule_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Returns ``(o, final_state)`` with ``o`` shaped ``[B, T, H, DV]``.

    ``g`` must be the forget gate in **log space** (same contract as
    ``vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule``), e.g. ``F.logsigmoid(...)``.

    If ``cu_seqlens`` is set, requires ``B == 1`` (packed batch). Segments are processed
    independently; ``initial_state`` must be ``[num_seg, H, DK, DV]`` or ``None`` (zeros).
    """
    if initial_state is not None:
        raise NotImplementedError(
            "TileLang pipeline in this patch does not yet fuse non-zero initial_state; "
            "use zeros for parity with Triton smoke tests."
        )

    if cu_seqlens is None:
        return _forward_dense(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=chunk_size,
        )

    if q.shape[0] != 1:
        raise ValueError("Varlen path expects batch size 1 (packed sequences).")
    num_seg = int(cu_seqlens.numel()) - 1
    outs: list[torch.Tensor] = []
    fss: list[torch.Tensor] = []
    cu = cu_seqlens.cpu().tolist()
    for i in range(num_seg):
        s0, s1 = int(cu[i]), int(cu[i + 1])
        qe = q[:, s0:s1]
        ke = k[:, s0:s1]
        ve = v[:, s0:s1]
        ge = g[:, s0:s1]
        be = beta[:, s0:s1]
        o_e, fs_e = _forward_dense(
            qe,
            ke,
            ve,
            ge,
            be,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=chunk_size,
        )
        outs.append(o_e)
        fss.append(fs_e)
    o_full = torch.cat(outs, dim=1)
    fs_full = torch.stack(fss, dim=0)
    return o_full, fs_full


def _forward_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must share [B, T, H].")
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    if g.shape != (B, T, H) or beta.shape != (B, T, H):
        raise ValueError("g and beta must be [B, T, H].")

    if use_qk_l2norm_in_kernel:
        qn = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
        kn = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
    else:
        qn, kn = q, k

    qh = qn.to(torch.float16)
    kh = kn.to(torch.float16)
    vh = v.to(torch.float16)
    gh = g.to(torch.float32)
    bh = beta.to(torch.float16)

    o, fs = run_opt_gdn_tilelang_pipeline(qh, kh, vh, gh, bh, chunk_size=chunk_size)
    return o, fs
