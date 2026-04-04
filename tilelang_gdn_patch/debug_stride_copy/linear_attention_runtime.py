"""
PyTorch entrypoints, reference, and test vectors for ``linear_attention_kernel_dump``.

Self-contained under ``debug_stride_copy`` (no imports from other project trees).
When extending coverage, update ``LINEAR_ATTENTION_TEST_CONFIGS`` / ``linear_attention_atol``
alongside any parallel script you maintain elsewhere.
"""
from __future__ import annotations

import torch

from linear_attention_kernel_dump import compiled_linear_attention_ker


def linear_attention(q, k, v, C):
    _, _, H, D = q.shape
    before = compiled_linear_attention_ker.cache_info().hits
    ker = compiled_linear_attention_ker(H, D, C)
    if compiled_linear_attention_ker.cache_info().hits > before:
        print(f"  reuse compiled kernel (H={H}, D={D}, C={C})")
    else:
        print(f"  !!recompile (H={H}, D={D}, C={C})")
    return ker(q, k, v)


def linear_attention_atol(L: int) -> float:
    """Absolute tolerance for ``assert_close`` vs sequence length (error accumulates)."""
    if L >= 4096:
        return 4e-2
    if L >= 2048:
        return 2e-2
    return 1e-2


# (B, H, L, D, C)
LINEAR_ATTENTION_TEST_CONFIGS = [
    (1, 2, 64, 128, 64),
    (1, 2, 256, 128, 64),
    (4, 2, 128, 128, 64),
    (8, 2, 512, 128, 64),
    (12, 2, 512, 128, 64),
    (16, 2, 256, 128, 64),
    (32, 2, 128, 128, 64),
    (50, 20, 128, 128, 64),
    (1, 2, 1024, 128, 64),
    (8, 2, 2048, 128, 64),
    (2, 2, 4096, 128, 64),
    (16, 2, 1024, 128, 64),
]


def ref_linear_attention(q, k, v):
    B, L, H, D = q.shape
    q = q.float()
    k = k.float()
    v = v.float()
    h = torch.zeros([B, H, D, D]).npu().to(torch.float)
    o = torch.zeros([B, L, H, D]).npu().to(torch.float)
    for i in range(L):
        q_i = q[:, i, :, :]
        k_i = k[:, i, :, :]
        v_i = v[:, i, :, :]
        dh = torch.einsum("bhi,bhj->bhij", k_i, v_i)
        h = h + dh
        o_i = torch.einsum("bhi,bhij->bhj", q_i, h)
        o[:, i, :, :] = o_i
    return o.to(torch.float16)
