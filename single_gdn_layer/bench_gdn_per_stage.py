#!/usr/bin/env python3
"""
Per-stage NPU micro-benchmarks for AscendQwen3_5GatedDeltaNet prefill (random weights).

Matches ``bench_gdn_forward_random.py`` setup (shapes, metadata, patched kernels).
Times: input projections, ``npu_causal_conv1d_custom``, ``fused_gdn_gating_patch``,
``chunk_gated_delta_rule``, and output (gated norm + out_proj).

Run on-device and paste results into ``gdn_per_stage.md`` (or use ``--markdown``).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
from einops import rearrange

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Reuse config helpers from the random GDN benchmark
from bench_gdn_forward_random import (  # type: ignore
    QWEN35_0_8B_CONFIG,
    QWEN35_9B_CONFIG,
    _init_dist_hccl,
    _parse_shapes,
    _randomize_gdn_layer,
    _write_config_temp,
)


def to_int64_tuple(t: torch.Tensor) -> tuple[int, ...]:
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (int(t.item()),)
    return tuple(int(x) for x in t.tolist())


def median_ms(warmup: int, repeats: int, fn) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times))


def chunk_analytical_flops(num_tokens: int, nv: int, dk: int, dv: int) -> float:
    """Same proxy as bench_gdn_forward*.py README: 32 * T * Nv * Dk * Dv."""
    return 32.0 * float(num_tokens * nv * dk * dv)


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def chunk_traffic_bytes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    o: torch.Tensor,
) -> int:
    """Read+write volume for chunk stage (inputs + output tensor sizes)."""
    return (
        tensor_bytes(q)
        + tensor_bytes(k)
        + tensor_bytes(v)
        + tensor_bytes(g)
        + tensor_bytes(beta)
        + tensor_bytes(initial_state)
        + tensor_bytes(o)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["0.8B", "9B"], default="0.8B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--shapes", nargs="+", default=["1x4096"], metavar="BxT")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a gdn_per_stage.md fragment to stdout",
    )
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    cfg_dict = QWEN35_0_8B_CONFIG if args.variant == "0.8B" else QWEN35_9B_CONFIG
    tmp_model_dir = _write_config_temp(cfg_dict)

    import torch_npu

    torch_npu.npu.set_device(args.device)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config

    engine_args = EngineArgs(
        model=tmp_model_dir,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        _init_dist_hccl(local_rank=args.device)

        import vllm_ascend.patch.worker  # noqa: F401 — patches chunk_gated_delta_rule etc.

        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
        from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch

        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device("npu", args.device)
        hf_text = vllm_config.model_config.hf_text_config

        vllm_config.compilation_config.static_forward_context.clear()

        layer_prefix = f"language_model.model.layers.{args.layer}.linear_attn"
        layer = Qwen3_5GatedDeltaNet(
            hf_text,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            speculative_config=vllm_config.speculative_config,
            prefix=layer_prefix,
        )
        layer = layer.to(device=device, dtype=dtype)
        _randomize_gdn_layer(layer, float(hf_text.initializer_range), args.seed)

        spec = layer.get_kv_cache_spec(vllm_config)
        assert spec is not None
        conv_shape, ssm_shape = spec.shapes
        conv_dt, ssm_dt = spec.dtypes
        num_blocks = 256
        conv_state = torch.zeros((num_blocks, *conv_shape), dtype=conv_dt, device=device)
        ssm_state = torch.zeros((num_blocks, *ssm_shape), dtype=ssm_dt, device=device)
        layer.kv_cache = [(conv_state, ssm_state)]

        builder = GDNAttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=[layer.prefix],
            vllm_config=vllm_config,
            device=device,
        )

        tp = layer.tp_size
        nv = hf_text.linear_num_value_heads // tp
        dk = hf_text.linear_key_head_dim
        dv = hf_text.linear_value_head_dim
        hidden_size = hf_text.hidden_size

        shapes = _parse_shapes(args.shapes)
        md_lines: list[str] = []

        print(
            f"[per-stage] variant={args.variant} npu={args.device} warmup={args.warmup} repeats={args.repeats}",
            flush=True,
        )

        for batch, seq_len in shapes:
            num_tokens = batch * seq_len
            query_lens = torch.tensor([seq_len] * batch, dtype=torch.int32, device="cpu")
            q_cpu = torch.zeros(batch + 1, dtype=torch.int32, device="cpu")
            q_cpu[1:] = torch.cumsum(query_lens, dim=0)
            q = q_cpu.to(device=device, non_blocking=True)
            seq_lens = query_lens.to(device=device, non_blocking=True)
            block_table = torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1)

            common = CommonAttentionMetadata(
                query_start_loc=q,
                query_start_loc_cpu=q_cpu,
                seq_lens=seq_lens,
                num_reqs=batch,
                num_actual_tokens=num_tokens,
                max_query_len=seq_len,
                max_seq_len=seq_len,
                block_table_tensor=block_table,
                slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            )

            meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
            attn_dict = {layer.prefix: meta}
            prebuilt = getattr(meta, "non_spec_chunked_prefill_meta", None)

            torch.manual_seed(args.seed + num_tokens)
            hidden_states = (
                torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
                * float(hf_text.initializer_range)
            )
            out_buf = torch.empty(num_tokens, hidden_size, device=device, dtype=dtype)

            # --- Reference tensors (one dry run, shapes for standalone stages) ---
            mixed_qkvz, _ = layer.in_proj_qkvz(hidden_states)
            qkv_size = (layer.key_dim * 2 + layer.value_dim) // tp
            z_size = layer.value_dim // tp
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, layer.head_v_dim)
            ba, _ = layer.in_proj_ba(hidden_states)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()

            conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if layer.activation else 0
            self_kv = layer.kv_cache[0]
            conv_tensor = self_kv[0]
            ssm_tensor = self_kv[1]

            non_spec_query_start_loc = meta.non_spec_query_start_loc
            non_spec_state_indices_tensor = meta.non_spec_state_indices_tensor
            has_initial_state = meta.has_initial_state

            with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                mixed_after_conv = torch.ops._C_ascend.npu_causal_conv1d_custom(
                    mixed_qkv.clone(),
                    conv_weights_T,
                    conv_state=conv_tensor,
                    bias_opt=layer.conv1d.bias,
                    query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc),
                    cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor),
                    initial_state_mode_opt=to_int64_tuple(has_initial_state),
                    num_accepted_tokens_opt=[],
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )

            query_ns, key_ns, value_ns = layer.rearrange_mixed_qkv(mixed_after_conv)
            g_ns, beta_ns = fused_gdn_gating_patch(layer.A_log, a, b, layer.dt_bias)
            initial_state = ssm_tensor[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0

            # Restore clean KV like a fresh prefill (match whole-layer benchmark first step)
            conv_tensor.zero_()
            ssm_tensor.zero_()

            # Standalone fake tensors (same shapes / dtypes as reference)
            q_f = torch.randn_like(query_ns)
            k_f = torch.randn_like(key_ns)
            v_f = torch.randn_like(value_ns)
            g_f = torch.randn_like(g_ns)
            beta_f = torch.randn_like(beta_ns)
            h0_f = torch.randn_like(initial_state)
            chunk_out_f = torch.empty_like(v_f)

            flops_chunk = chunk_analytical_flops(num_tokens, nv, dk, dv)
            bytes_chunk = chunk_traffic_bytes(q_f, k_f, v_f, g_f, beta_f, h0_f, chunk_out_f)

            def full_forward():
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    layer.forward(hidden_states, out_buf)

            def stage_input_proj():
                mixed_qkvz2, _ = layer.in_proj_qkvz(hidden_states)
                _mq, _z = mixed_qkvz2.split([qkv_size, z_size], dim=-1)
                _z = _z.reshape(_z.size(0), -1, layer.head_v_dim)
                ba2, _ = layer.in_proj_ba(hidden_states)
                b2, a2 = ba2.chunk(2, dim=-1)
                _ = b2.contiguous()
                _ = a2.contiguous()

            def stage_output_proj():
                core = torch.randn(
                    num_tokens,
                    layer.num_v_heads // tp,
                    layer.head_v_dim,
                    device=device,
                    dtype=dtype,
                )
                z3 = torch.randn_like(z)
                z_shape_og = z3.shape
                co = core.reshape(-1, core.shape[-1])
                zz = z3.reshape(-1, z3.shape[-1])
                co = layer.norm(co, zz)
                co = co.reshape(z_shape_og)
                co = rearrange(co, "... h d -> ... (h d)")
                o2, _ = layer.out_proj(co)

            def stage_conv():
                conv_tensor.zero_()
                ssm_tensor.zero_()
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv.clone(),
                        conv_weights_T,
                        conv_state=conv_tensor,
                        bias_opt=layer.conv1d.bias,
                        query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc),
                        cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor),
                        initial_state_mode_opt=to_int64_tuple(has_initial_state),
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )

            def stage_gating():
                fused_gdn_gating_patch(layer.A_log, a, b, layer.dt_bias)

            def stage_chunk_fake():
                h0 = h0_f.clone()
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    chunk_gated_delta_rule(
                        q=q_f,
                        k=k_f,
                        v=v_f,
                        g=g_f,
                        beta=beta_f,
                        initial_state=h0,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=prebuilt,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )

            def stage_chunk_real():
                h0r = initial_state.clone()
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    chunk_gated_delta_rule(
                        q=query_ns,
                        k=key_ns,
                        v=value_ns,
                        g=g_ns,
                        beta=beta_ns,
                        initial_state=h0r,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=prebuilt,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )

            t_full = median_ms(args.warmup, args.repeats, full_forward)
            t_in = median_ms(args.warmup, args.repeats, stage_input_proj)
            t_out = median_ms(args.warmup, args.repeats, stage_output_proj)
            t_conv = median_ms(args.warmup, args.repeats, stage_conv)
            t_gate = median_ms(args.warmup, args.repeats, stage_gating)
            t_chunk_fake = median_ms(args.warmup, args.repeats, stage_chunk_fake)
            t_chunk_real = median_ms(args.warmup, args.repeats, stage_chunk_real)

            s = t_full / 1000.0
            tflops_chunk = flops_chunk / (t_chunk_real / 1000.0) / 1e12
            gib_s_chunk = bytes_chunk / (t_chunk_real / 1000.0) / (1024**3)

            # Ratios vs full forward (standalone times are not expected to sum to t_full)
            r_in = 100.0 * t_in / t_full
            r_out = 100.0 * t_out / t_full
            r_conv = 100.0 * t_conv / t_full
            r_gate = 100.0 * t_gate / t_full
            r_chunk = 100.0 * t_chunk_real / t_full

            print(
                f"\nshape batch×seq={batch}×{seq_len} (T={num_tokens}) | full_forward median {t_full:.3f} ms",
                flush=True,
            )
            print(
                f"  input_proj          {t_in:8.3f} ms  ({r_in:5.1f}% of full)",
                flush=True,
            )
            print(
                f"  causal_conv (npu)   {t_conv:8.3f} ms  ({r_conv:5.1f}% of full)",
                flush=True,
            )
            print(
                f"  fused_gdn_gating    {t_gate:8.3f} ms  ({r_gate:5.1f}% of full)",
                flush=True,
            )
            print(
                f"  chunk_gdr (real I/O){t_chunk_real:8.3f} ms  ({r_chunk:5.1f}% of full)",
                flush=True,
            )
            print(
                f"  chunk_gdr (fake I/O){t_chunk_fake:8.3f} ms  (sanity vs real)",
                flush=True,
            )
            print(
                f"  output norm+proj    {t_out:8.3f} ms  ({r_out:5.1f}% of full)",
                flush=True,
            )
            print(
                f"  chunk TFLOP/s (proxy {flops_chunk:.3e} FLOP)  {tflops_chunk:.2f}",
                flush=True,
            )
            print(
                f"  chunk GiB/s (tensor traffic {bytes_chunk} B model) {gib_s_chunk:.2f}",
                flush=True,
            )

            md_lines.append(
                f"| {batch}×{seq_len} (T={num_tokens}) | {t_full:.3f} | {t_in:.3f} | {t_conv:.3f} | "
                f"{t_gate:.3f} | {t_chunk_real:.3f} | {t_out:.3f} | {r_in:.1f}% | {r_conv:.1f}% | "
                f"{r_gate:.1f}% | {r_chunk:.1f}% | {r_out:.1f}% | {tflops_chunk:.2f} | {gib_s_chunk:.2f} |"
            )

        destroy_model_parallel()
        destroy_distributed_environment()

    if args.markdown:
        print("\n--- markdown table ---\n")
        header = (
            "| batch×seq (T) | full (ms) | in (ms) | conv (ms) | gate (ms) | chunk (ms) | out (ms) | "
            "%in | %conv | %gate | %chunk | %out | chunk TFLOP/s | chunk GiB/s |"
        )
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        print(header)
        print(sep)
        for line in md_lines:
            print(line)


if __name__ == "__main__":
    main()
