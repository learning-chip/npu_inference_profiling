"""
NPU correctness: TileLang linear attention vs ``ref_linear_attention``.

Imports only from this directory (``linear_attention_runtime``).

Requires Ascend NPU + working tilelang; skips when ``torch.npu`` is unavailable.
"""
from __future__ import annotations

import pytest
import torch

from linear_attention_runtime import (
    LINEAR_ATTENTION_TEST_CONFIGS,
    linear_attention,
    linear_attention_atol,
    ref_linear_attention,
)


def _npu_ready() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.parametrize("B,H,L,D,C", LINEAR_ATTENTION_TEST_CONFIGS)
def test_linear_attention_matches_reference(B: int, H: int, L: int, D: int, C: int) -> None:
    if not _npu_ready():
        pytest.skip("Ascend NPU not available (torch.npu)")

    torch.manual_seed(0)
    q = torch.randn([B, L, H, D]).npu().to(torch.float16)
    k = torch.randn([B, L, H, D]).npu().to(torch.float16)
    v = torch.randn([B, L, H, D]).npu().to(torch.float16)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)

    o = linear_attention(q, k, v, C)
    ref_o = ref_linear_attention(q, k, v)

    torch.testing.assert_close(
        o.cpu(),
        ref_o.cpu(),
        rtol=1e-2,
        atol=linear_attention_atol(L),
    )
