"""
Minimal linear-attention TileLang kernel for AscendC dumps (``[B, L, H, D]``).

Used only inside ``debug_stride_copy`` (``dump_kernel.py``, ``linear_attention_runtime.py``).
"""
import functools

import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()
tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

try:
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
except Exception:
    core_num = 24


@tilelang.jit(out_idx=[-1], workspace_idx=[3, 4], pass_configs=pass_configs)
def linear_attention_ker(H, D, C, dtype="float16", accum_dtype="float"):
    B = T.symbolic("B")
    L = T.symbolic("L")

    shape = [B, L, H, D]
    chunk_num = T.ceildiv(L, C)
    VEC_NUM = 2

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        workspace_1: T.Tensor([core_num, C, C], dtype),
        workspace_2: T.Tensor([core_num, D, D], dtype),
        O: T.Tensor(shape, dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([C, D], dtype)
            k_l1 = T.alloc_L1([C, D], dtype)
            v_l1 = T.alloc_L1([C, D], dtype)
            h_l1 = T.alloc_L1([D, D], dtype)
            acc_l1 = T.alloc_L1([C, C], dtype)
            h_l0 = T.alloc_L0C([D, D], accum_dtype)
            acc_l0 = T.alloc_L0C([C, C], accum_dtype)
            o_l0 = T.alloc_L0C([C, D], accum_dtype)

            hsum_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
            h_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
            acc_ub = T.alloc_ub([C // VEC_NUM, C], dtype)
            zero_ub = T.alloc_ub([C // VEC_NUM, C], dtype)

            with T.Scope("C"):
                for work_idx in T.serial(T.ceildiv(B * H, core_num)):
                    pid = work_idx * core_num + cid
                    if pid < B * H:
                        by = pid % H
                        bz = pid // H

                        T.wait_cross_flag(1)

                        for i in T.serial(chunk_num):
                            T.copy(Q[bz, i * C, by, 0], q_l1)
                            T.copy(K[bz, i * C, by, 0], k_l1)
                            T.copy(V[bz, i * C, by, 0], v_l1)
                            T.copy(workspace_2[cid, 0, 0], h_l1)
                            T.gemm_v0(q_l1, k_l1, acc_l0, transpose_B=True, init=True)
                            T.copy(acc_l0, workspace_1[cid, 0, 0])
                            T.gemm_v0(k_l1, v_l1, h_l0, transpose_A=True, init=True)
                            T.copy(h_l0, workspace_2[cid, 0, 0])
                            T.set_cross_flag("FIX", 0)

                            T.wait_cross_flag(1)
                            T.copy(workspace_1[cid, 0, 0], acc_l1)
                            T.gemm_v0(acc_l1, v_l1, o_l0, init=True)
                            T.gemm_v0(q_l1, h_l1, o_l0, init=False)
                            T.copy(o_l0, O[bz, i * C, by, 0])

            with T.Scope("V"):
                T.tile.fill(zero_ub, 0.0)

                for work_idx in T.serial(T.ceildiv(B * H, core_num)):
                    pid = work_idx * core_num + cid
                    if pid < B * H:
                        T.tile.fill(hsum_ub, 0.0)
                        T.copy(hsum_ub, workspace_2[cid, vid * D // VEC_NUM, 0])
                        T.set_cross_flag("MTE3", 1)

                        for _i in T.serial(chunk_num):
                            T.wait_cross_flag(0)
                            T.copy(workspace_1[cid, vid * C // VEC_NUM, 0], acc_ub)
                            T.copy(workspace_2[cid, vid * D // VEC_NUM, 0], h_ub)
                            for j in range(C // VEC_NUM):
                                for k in range(C):
                                    if (j + vid * C // VEC_NUM) < k:
                                        acc_ub[j, k] = zero_ub[j, k]
                            T.tile.add(hsum_ub, hsum_ub, h_ub)
                            T.copy(acc_ub, workspace_1[cid, vid * C // VEC_NUM, 0])
                            T.copy(hsum_ub, workspace_2[cid, vid * D // VEC_NUM, 0])
                            T.set_cross_flag("MTE3", 1)

    return main


@functools.lru_cache(maxsize=None)
def compiled_linear_attention_ker(H, D, C):
    return linear_attention_ker(H, D, C)
