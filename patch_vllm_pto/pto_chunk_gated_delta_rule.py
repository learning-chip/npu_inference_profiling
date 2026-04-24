# SPDX-License-Identifier: Apache-2.0
"""PTO-backed ``chunk_gated_delta_rule`` (C=128) for vLLM-Ascend Qwen3.5 prefill.

Falls back to the Ascend Triton implementation when PTO cannot match semantics
(non-zero ``initial_state``, PCP multi-device, or missing NPU).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from einops import rearrange

if TYPE_CHECKING:
    pass

C_PTO = 128

# Repo paths (stable under /workdir in this environment)
_PTO_KERNELS = "/workdir/pto-kernels/examples/jit_cpp"
_CHUNK_GDN_DYN = os.path.join(_PTO_KERNELS, "chunk_gdn", "dynamic_bsnd")
_FAST_INV = os.path.join(_PTO_KERNELS, "fast_inverse")


def _ensure_pto_sys_path() -> None:
    for p in (_CHUNK_GDN_DYN, _FAST_INV):
        if p not in sys.path:
            sys.path.insert(0, p)


@lru_cache(maxsize=1)
def _tri_inv_kernel():
    _ensure_pto_sys_path()
    from jit_util_fast_inverse import jit_compile  # type: ignore

    cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
    return jit_compile(cpp, verbose=False)


def _count_varlen_chunks(cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    cu = cu_seqlens.detach().cpu().tolist()
    return sum(
        (int(eos) - int(bos) + chunk_size - 1) // chunk_size
        for bos, eos in zip(cu[:-1], cu[1:], strict=False)
    )


def _make_minus_identity(matrix_size: int, device: torch.device) -> torch.Tensor:
    minus_identity = torch.zeros((matrix_size, matrix_size), dtype=torch.float16, device=device)
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def pto_solve_tril(
    tri_inv_func,
    A_fp16: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    num_heads: int,
) -> torch.Tensor:
    num_matrices = _count_varlen_chunks(cu_seqlens, chunk_size) * num_heads
    tensor_out = torch.zeros_like(A_fp16, dtype=torch.float32)
    minus_identity = _make_minus_identity(chunk_size, A_fp16.device)
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    from dynamic_kernel_libs import BLOCK_DIM  # type: ignore

    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        A_fp16,
        minus_identity,
        chunk_size,
        num_matrices,
        num_heads,
        cu_seqlens=cu32,
        block_dim=BLOCK_DIM,
        is_lower=True,
    )
    torch.npu.synchronize()
    return tensor_out.to(torch.float16)


def _needs_triton_fallback(
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    if initial_state is not None and torch.any(initial_state != 0):
        return True
    if cu_seqlens is None:
        return True
    return False


def _pto_forward_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns (o, final_state) with same dtype/device as ``q`` (final_state fp16/bf16)."""
    _ensure_pto_sys_path()
    from dynamic_kernel_libs import (  # type: ignore
        BLOCK_DIM,
        run_chunk_cumsum,
        run_chunk_h,
        run_chunk_o,
        run_scaled_dot_kkt,
        run_wy_fast,
        total_chunks,
    )

    dev = q.device
    out_dtype = q.dtype
    N_seq = int(cu_seqlens.numel() - 1)
    T = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]

    # PTO stack expects fp16 activations; accumulate g in fp32
    q_w = q.to(torch.float16)
    k_w = k.to(torch.float16)
    v_w = v.to(torch.float16)
    beta_w = beta.to(torch.float16)
    g_w = g.to(torch.float32) if g.dtype != torch.float32 else g

    cu32 = cu_seqlens.to(torch.int32).contiguous()

    msk_lower = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()

    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    with torch.autograd.profiler.record_function("PTO_gdn_chunk_cumsum"):
        run_chunk_cumsum(
            g_w,
            g_sum,
            chunk_size=C_PTO,
            cu_seqlens=cu32,
            batch_size_override=N_seq,
        )

    A_out = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    with torch.autograd.profiler.record_function("PTO_gdn_scaled_dot_kkt"):
        run_scaled_dot_kkt(
            k_w,
            beta_w,
            g_sum,
            msk_lower,
            None,
            A_out,
            chunk_size=C_PTO,
            cu_seqlens=cu32,
            batch_size_override=N_seq,
        )

    with torch.autograd.profiler.record_function("PTO_gdn_solve_tril"):
        A_sol = pto_solve_tril(_tri_inv_kernel(), A_out, cu32, C_PTO, H)

    w_out = torch.empty_like(k_w)
    u_out = torch.empty_like(v_w)
    with torch.autograd.profiler.record_function("PTO_gdn_wy_fast"):
        run_wy_fast(
            k_w,
            v_w,
            beta_w,
            g_sum,
            A_sol,
            w_out,
            u_out,
            chunk_size=C_PTO,
            cu_seqlens=cu32,
            batch_size_override=N_seq,
        )

    tc_n = total_chunks(N_seq, T, C_PTO, cu32)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v_w)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    with torch.autograd.profiler.record_function("PTO_gdn_chunk_h"):
        run_chunk_h(
            k_w,
            w_out,
            u_out,
            g_sum,
            s_out,
            v_new,
            fs_out,
            chunk_size=C_PTO,
            cu_seqlens=cu32,
            batch_size_override=N_seq,
        )

    o_fp16 = torch.empty_like(q_w)
    with torch.autograd.profiler.record_function("PTO_gdn_chunk_o"):
        run_chunk_o(
            q_w,
            k_w,
            v_new,
            s_out,
            g_sum,
            msk_full,
            o_fp16,
            chunk_size=C_PTO,
            cu_seqlens=cu32,
            batch_size_override=N_seq,
        )

    o = (o_fp16 * scale).to(out_dtype)
    final = fs_out.view(N_seq, H, D, D).to(out_dtype)
    return o, final


@torch.compiler.disable
def chunk_gated_delta_rule_pto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    _triton_impl,
):
    """Drop-in replacement for ``vllm_ascend...chunk_gated_delta_rule``."""
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, (
        "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    )

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2,
        )
        q, k, v, beta, g = map(lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        import warnings

        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                "Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd

        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    try:
        from vllm.distributed import get_pcp_group

        if get_pcp_group().world_size > 1:
            return _triton_impl(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                prebuilt_meta=prebuilt_meta,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
    except Exception:
        pass

    if q.device.type != "npu":
        return _triton_impl(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )

    if _needs_triton_fallback(initial_state, cu_seqlens):
        return _triton_impl(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )

    o, final_state = _pto_forward_core(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        float(scale),
    )
    if not output_final_state:
        final_state = None
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state


def bind_triton(_triton_impl):
    """Partial application: returns a callable matching the public vLLM API."""

    def _bound(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        prebuilt_meta=None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        return chunk_gated_delta_rule_pto(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            _triton_impl=_triton_impl,
        )

    _bound.__name__ = "chunk_gated_delta_rule"
    _bound.__doc__ = chunk_gated_delta_rule_pto.__doc__
    # Introspection hook for tests / compare scripts (Ascend Triton op has no such attr).
    _bound._vllm_pto_chunk_wrapper_installed = True
    return _bound
