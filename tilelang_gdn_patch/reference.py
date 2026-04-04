"""
Sequential GDN reference (same as ``opt_gdn_full.py::ref_seq_gdn``), for numerical checks.
"""

from __future__ import annotations

import torch


def ref_seq_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Head-first layout ``[B, H, L, DK]``, ``v`` ``[B, H, L, DV]``; ``g`` log-space."""
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    batch, h, l, dk = q.shape
    dv = v.shape[-1]
    s = torch.zeros((batch, h, dv, dk), device=q.device, dtype=torch.float32)
    o = torch.empty((batch, h, l, dv), device=q.device, dtype=torch.float32)
    i = torch.eye(dk, device=q.device, dtype=torch.float32).view(1, 1, dk, dk)
    for t in range(l):
        q_i = q[:, :, t, :]
        k_i = k[:, :, t, :]
        v_i = v[:, :, t, :]
        beta_i = beta[:, :, t].view(batch, h, 1, 1)
        g_i = g[:, :, t].view(batch, h, 1, 1)
        kkt = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        vkt = v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        a_i = g_i * (i - beta_i * kkt)
        term_1 = torch.matmul(s, a_i)
        term_2 = beta_i * vkt
        s = term_1 + term_2
        o[:, :, t, :] = torch.einsum("bhpq,bhq->bhp", s, q_i)
    return o.to(torch.float16)


def ref_seq_gdn_bth(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Same recurrence as :func:`ref_seq_gdn`, but tensors are Triton layout
    ``[B, T, H, DK]`` / ``[B, T, H, DV]`` and ``g`` is log-space.
    """
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    b, seq, h, dk = q.shape
    dv = v.shape[-1]
    s = torch.zeros((b, h, dv, dk), device=q.device, dtype=torch.float32)
    o = torch.empty((b, seq, h, dv), device=q.device, dtype=torch.float32)
    i = torch.eye(dk, device=q.device, dtype=torch.float32).view(1, 1, dk, dk)
    for t in range(seq):
        q_i = q[:, t, :, :]
        k_i = k[:, t, :, :]
        v_i = v[:, t, :, :]
        beta_i = beta[:, t, :].view(b, h, 1, 1)
        g_i = g[:, t, :].view(b, h, 1, 1)
        kkt = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        vkt = v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        a_i = g_i * (i - beta_i * kkt)
        term_1 = torch.matmul(s, a_i)
        term_2 = beta_i * vkt
        s = term_1 + term_2
        o[:, t, :, :] = torch.einsum("bhpq,bhq->bhp", s, q_i)
    return o.to(torch.float16)
