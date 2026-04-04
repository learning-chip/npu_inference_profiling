# Qwen3.5 prefill profiling sweep

This report is generated from `run_qwen35_profile_sweep.py` using Ascend 
`op_statistic.csv` (top operators by total-time share).

## End-to-end run times

| Model | Mode | Batch | Seq len | Status | Time (s) | Notes |
|---|:---:|---:|---:|:---:|---:|---|
| 9B | eager | 1 | 4096 | ok | 124.74 |  |
| 9B | eager | 1 | 16384 | ok | 129.49 |  |
| 9B | eager | 2 | 8192 | ok | 132.36 |  |
| 9B | eager | 16 | 1024 | ok | 134.83 |  |
| 0.8B | eager | 1 | 4096 | ok | 98.81 |  |
| 0.8B | eager | 1 | 16384 | ok | 99.92 |  |
| 0.8B | eager | 2 | 8192 | ok | 100.75 |  |
| 0.8B | eager | 16 | 1024 | ok | 100.85 |  |

## Top-10 operators by case (from op_statistic.csv)

### qwen35_9B_eager_b1_sl4K (9B, eager, batch=1, seq=4096)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | MatMulV3 | 53.462 |
| 2 | CausalConv1d | 9.916 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 7.772 |
| 4 | recompute_w_u_fwd_kernel | 4.793 |
| 5 | chunk_fwd_kernel_o | 4.141 |
| 6 | merge_16x16_to_64x64_inverse_kernel | 3.371 |
| 7 | AddRmsNormBias | 2.211 |
| 8 | SwiGlu | 2.065 |
| 9 | FusedInferAttentionScore | 1.552 |
| 10 | _layer_norm_fwd_1pass_kernel_npu | 1.477 |

### qwen35_9B_eager_b1_sl16K (9B, eager, batch=1, seq=16384)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | MatMulV3 | 50.304 |
| 2 | CausalConv1d | 10.268 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 7.802 |
| 4 | FusedInferAttentionScore | 5.475 |
| 5 | recompute_w_u_fwd_kernel | 4.783 |
| 6 | chunk_fwd_kernel_o | 4.223 |
| 7 | merge_16x16_to_64x64_inverse_kernel | 3.506 |
| 8 | AddRmsNormBias | 2.421 |
| 9 | SwiGlu | 2.322 |
| 10 | Slice | 1.883 |

### qwen35_9B_eager_b2_sl8K (9B, eager, batch=2, seq=8192)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | MatMulV3 | 52.135 |
| 2 | CausalConv1d | 10.406 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 8.028 |
| 4 | recompute_w_u_fwd_kernel | 4.932 |
| 5 | chunk_fwd_kernel_o | 4.334 |
| 6 | merge_16x16_to_64x64_inverse_kernel | 3.444 |
| 7 | FusedInferAttentionScore | 2.899 |
| 8 | SwiGlu | 2.343 |
| 9 | AddRmsNormBias | 2.314 |
| 10 | _layer_norm_fwd_1pass_kernel_npu | 1.557 |

### qwen35_9B_eager_b16_sl1K (9B, eager, batch=16, seq=1024)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | MatMulV3 | 52.509 |
| 2 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 7.152 |
| 3 | recompute_w_u_fwd_kernel | 5.159 |
| 4 | MatMulV2 | 4.486 |
| 5 | CausalConv1d | 3.837 |
| 6 | merge_16x16_to_64x64_inverse_kernel | 3.735 |
| 7 | chunk_fwd_kernel_o | 3.616 |
| 8 | IndexPutV2 | 3.413 |
| 9 | AddRmsNormBias | 2.536 |
| 10 | SwiGlu | 2.466 |

### qwen35_0.8B_eager_b1_sl4K (0.8B, eager, batch=1, seq=4096)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | CausalConv1d | 22.767 |
| 2 | MatMulV3 | 15.677 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 10.754 |
| 4 | recompute_w_u_fwd_kernel | 7.764 |
| 5 | merge_16x16_to_64x64_inverse_kernel | 5.622 |
| 6 | chunk_fwd_kernel_o | 5.576 |
| 7 | MatMulV2 | 2.787 |
| 8 | FusedInferAttentionScore | 2.603 |
| 9 | Slice | 2.588 |
| 10 | solve_tril_16x16_kernel | 2.520 |

### qwen35_0.8B_eager_b1_sl16K (0.8B, eager, batch=1, seq=16384)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | CausalConv1d | 23.788 |
| 2 | MatMulV3 | 10.883 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 10.749 |
| 4 | FusedInferAttentionScore | 8.634 |
| 5 | recompute_w_u_fwd_kernel | 7.713 |
| 6 | chunk_fwd_kernel_o | 6.405 |
| 7 | merge_16x16_to_64x64_inverse_kernel | 5.895 |
| 8 | MatMulV2 | 5.328 |
| 9 | Slice | 4.257 |
| 10 | _layer_norm_fwd_1pass_kernel_npu | 2.469 |

### qwen35_0.8B_eager_b2_sl8K (0.8B, eager, batch=2, seq=8192)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | CausalConv1d | 24.688 |
| 2 | MatMulV3 | 16.754 |
| 3 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 11.346 |
| 4 | recompute_w_u_fwd_kernel | 8.279 |
| 5 | chunk_fwd_kernel_o | 6.376 |
| 6 | merge_16x16_to_64x64_inverse_kernel | 5.722 |
| 7 | FusedInferAttentionScore | 4.850 |
| 8 | Slice | 2.666 |
| 9 | _layer_norm_fwd_1pass_kernel_npu | 2.559 |
| 10 | solve_tril_16x16_kernel | 1.983 |

### qwen35_0.8B_eager_b16_sl1K (0.8B, eager, batch=16, seq=1024)

| Rank | Operator | Ratio (%) |
|---:|:---|---:|
| 1 | MatMulV3 | 13.497 |
| 2 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 12.773 |
| 3 | recompute_w_u_fwd_kernel | 9.258 |
| 4 | CausalConv1d | 8.689 |
| 5 | MatMulV2 | 7.199 |
| 6 | merge_16x16_to_64x64_inverse_kernel | 6.622 |
| 7 | chunk_fwd_kernel_o | 6.426 |
| 8 | IndexPutV2 | 6.230 |
| 9 | Slice | 4.974 |
| 10 | _layer_norm_fwd_1pass_kernel_npu | 2.930 |

## How top operators shift with settings (data from this sweep)

### Sweep grid: top-1 operator per cell (eager, run order)

| Model | Batch | Seq len | Case id | Top-1 op | Ratio % | Status |
|---|:---:|---:|:---|:---|---:|:---|
| 9B | 1 | 4096 | `qwen35_9B_eager_b1_sl4K` | MatMulV3 | 53.462 | ok |
| 9B | 1 | 16384 | `qwen35_9B_eager_b1_sl16K` | MatMulV3 | 50.304 | ok |
| 9B | 2 | 8192 | `qwen35_9B_eager_b2_sl8K` | MatMulV3 | 52.135 | ok |
| 9B | 16 | 1024 | `qwen35_9B_eager_b16_sl1K` | MatMulV3 | 52.509 | ok |
| 0.8B | 1 | 4096 | `qwen35_0.8B_eager_b1_sl4K` | CausalConv1d | 22.767 | ok |
| 0.8B | 1 | 16384 | `qwen35_0.8B_eager_b1_sl16K` | CausalConv1d | 23.788 | ok |
| 0.8B | 2 | 8192 | `qwen35_0.8B_eager_b2_sl8K` | CausalConv1d | 24.688 | ok |
| 0.8B | 16 | 1024 | `qwen35_0.8B_eager_b16_sl1K` | MatMulV3 | 13.497 | ok |

### Most common operators among top-10 lists (by case count)

Ranked by how many successful cases include the operator in the top-10:


- **MatMulV3**: 8 case(s) in top-10 — e.g. qwen35_9B_eager_b1_sl4K (53.5%), qwen35_9B_eager_b16_sl1K (52.5%), qwen35_9B_eager_b2_sl8K (52.1%), qwen35_9B_eager_b1_sl16K (50.3%), qwen35_0.8B_eager_b2_sl8K (16.8%), … (+3 cases)
- **CausalConv1d**: 8 case(s) in top-10 — e.g. qwen35_0.8B_eager_b2_sl8K (24.7%), qwen35_0.8B_eager_b1_sl16K (23.8%), qwen35_0.8B_eager_b1_sl4K (22.8%), qwen35_9B_eager_b2_sl8K (10.4%), qwen35_9B_eager_b1_sl16K (10.3%), … (+3 cases)
- **chunk_gated_delta_rule_fwd_kernel_h_blockdim64**: 8 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (12.8%), qwen35_0.8B_eager_b2_sl8K (11.3%), qwen35_0.8B_eager_b1_sl4K (10.8%), qwen35_0.8B_eager_b1_sl16K (10.7%), qwen35_9B_eager_b2_sl8K (8.0%), … (+3 cases)
- **recompute_w_u_fwd_kernel**: 8 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (9.3%), qwen35_0.8B_eager_b2_sl8K (8.3%), qwen35_0.8B_eager_b1_sl4K (7.8%), qwen35_0.8B_eager_b1_sl16K (7.7%), qwen35_9B_eager_b16_sl1K (5.2%), … (+3 cases)
- **merge_16x16_to_64x64_inverse_kernel**: 8 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (6.6%), qwen35_0.8B_eager_b1_sl16K (5.9%), qwen35_0.8B_eager_b2_sl8K (5.7%), qwen35_0.8B_eager_b1_sl4K (5.6%), qwen35_9B_eager_b16_sl1K (3.7%), … (+3 cases)
- **chunk_fwd_kernel_o**: 8 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (6.4%), qwen35_0.8B_eager_b1_sl16K (6.4%), qwen35_0.8B_eager_b2_sl8K (6.4%), qwen35_0.8B_eager_b1_sl4K (5.6%), qwen35_9B_eager_b2_sl8K (4.3%), … (+3 cases)
- **FusedInferAttentionScore**: 6 case(s) in top-10 — e.g. qwen35_0.8B_eager_b1_sl16K (8.6%), qwen35_9B_eager_b1_sl16K (5.5%), qwen35_0.8B_eager_b2_sl8K (4.8%), qwen35_9B_eager_b2_sl8K (2.9%), qwen35_0.8B_eager_b1_sl4K (2.6%), … (+1 cases)
- **Slice**: 5 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (5.0%), qwen35_0.8B_eager_b1_sl16K (4.3%), qwen35_0.8B_eager_b2_sl8K (2.7%), qwen35_0.8B_eager_b1_sl4K (2.6%), qwen35_9B_eager_b1_sl16K (1.9%)
- **_layer_norm_fwd_1pass_kernel_npu**: 5 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (2.9%), qwen35_0.8B_eager_b2_sl8K (2.6%), qwen35_0.8B_eager_b1_sl16K (2.5%), qwen35_9B_eager_b2_sl8K (1.6%), qwen35_9B_eager_b1_sl4K (1.5%)
- **MatMulV2**: 4 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (7.2%), qwen35_0.8B_eager_b1_sl16K (5.3%), qwen35_9B_eager_b16_sl1K (4.5%), qwen35_0.8B_eager_b1_sl4K (2.8%)
- **AddRmsNormBias**: 4 case(s) in top-10 — e.g. qwen35_9B_eager_b16_sl1K (2.5%), qwen35_9B_eager_b1_sl16K (2.4%), qwen35_9B_eager_b2_sl8K (2.3%), qwen35_9B_eager_b1_sl4K (2.2%)
- **SwiGlu**: 4 case(s) in top-10 — e.g. qwen35_9B_eager_b16_sl1K (2.5%), qwen35_9B_eager_b2_sl8K (2.3%), qwen35_9B_eager_b1_sl16K (2.3%), qwen35_9B_eager_b1_sl4K (2.1%)
- **IndexPutV2**: 2 case(s) in top-10 — e.g. qwen35_0.8B_eager_b16_sl1K (6.2%), qwen35_9B_eager_b16_sl1K (3.4%)
- **solve_tril_16x16_kernel**: 2 case(s) in top-10 — e.g. qwen35_0.8B_eager_b1_sl4K (2.5%), qwen35_0.8B_eager_b2_sl8K (2.0%)

## Trace zip bundles

- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_9B_eager_b1_sl4K_ascend_profile.zip` — qwen35_9B_eager_b1_sl4K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_9B_eager_b1_sl16K_ascend_profile.zip` — qwen35_9B_eager_b1_sl16K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_9B_eager_b2_sl8K_ascend_profile.zip` — qwen35_9B_eager_b2_sl8K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_9B_eager_b16_sl1K_ascend_profile.zip` — qwen35_9B_eager_b16_sl1K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_0.8B_eager_b1_sl4K_ascend_profile.zip` — qwen35_0.8B_eager_b1_sl4K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_0.8B_eager_b1_sl16K_ascend_profile.zip` — qwen35_0.8B_eager_b1_sl16K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_0.8B_eager_b2_sl8K_ascend_profile.zip` — qwen35_0.8B_eager_b2_sl8K
- `/workdir/npu_inference_profiling/profile_sweep_runs/zips/qwen35_0.8B_eager_b16_sl1K_ascend_profile.zip` — qwen35_0.8B_eager_b16_sl1K

