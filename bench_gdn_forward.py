#!/usr/bin/env python3
"""
Standalone NPU benchmark for AscendQwen3_5GatedDeltaNet.forward (vLLM + vllm-ascend).

Loads real weights from a local Qwen3.5 snapshot, builds GDN prefill attention metadata
matching vLLM's scheduler (CommonAttentionMetadata + GDNAttentionMetadataBuilder), and
times the patched forward used during ``llm.generate`` prefill.

Reports measured latency (median of ``torch.npu.synchronize``-bounded timings), effective
TFLOP/s from an analytical model (GEMM FLOPs for projections + a fixed proxy for the
recurrent GDN core—labeled in output), and effective GiB/s from parameter bytes plus a
rough activation traffic estimate.

Prefill metadata matches vLLM V1 ``GDNAttentionMetadataBuilder`` (``CommonAttentionMetadata``
built like a cold prefill: ``seq_lens == query_lens``). Default shapes align with
``profile_qwen35_prefill.py`` (e.g. 4×4096 tokens).

Requires: NPU visible (see ``npu-smi info``), vLLM + vllm-ascend with Triton/GDN kernels,
and ``import vllm_ascend.vllm_ascend_C`` for Ascend causal-conv ops.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import torch

# -----------------------------------------------------------------------------
# Env (match typical vLLM-Ascend worker)
# -----------------------------------------------------------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _parse_shapes(specs: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in specs:
        if "x" not in s.lower():
            raise ValueError(f"Bad shape {s!r}, expected BATCHxSEQ_LEN e.g. 4x4096")
        a, b = s.lower().split("x", 1)
        out.append((int(a.strip()), int(b.strip())))
    return out


@dataclass(frozen=True)
class RooflineEstimates:
    """Rough roofline inputs; GDN core uses a fixed proxy multiplier (documented)."""

    flops_gemm: float
    flops_gdn_proxy: float
    bytes_weights: int
    bytes_activations: int

    @property
    def flops_total(self) -> float:
        return self.flops_gemm + self.flops_gdn_proxy

    @property
    def bytes_total(self) -> int:
        return self.bytes_weights + self.bytes_activations


def _estimate_roofline(layer, num_tokens: int, hidden: int, cfg, device: torch.device) -> RooflineEstimates:
    """Analytical FLOPs (GEMM) + simple proxy for recurrent GDN core; byte traffic model."""
    tp = layer.tp_size
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    # in_proj_qkvz: [key, key, value, value] packed
    out_qkvz = 2 * key_dim + 2 * value_dim
    flops_qkvz = 2 * num_tokens * hidden * (out_qkvz // tp)
    # in_proj_ba: two shards of num_v_heads
    nh = cfg.linear_num_value_heads
    out_ba = 2 * (nh // tp)
    flops_ba = 2 * num_tokens * hidden * out_ba
    flops_out = 2 * num_tokens * (value_dim // tp) * hidden

    # conv1d ColumnParallel: approximate as batched GEMM along conv dim (upper bound)
    conv_dim = key_dim * 2 + value_dim
    k = cfg.linear_conv_kernel_dim
    flops_conv = 2 * num_tokens * conv_dim * k  # order-of-magnitude

    flops_gemm = flops_qkvz + flops_ba + flops_out + flops_conv

    # Proxy for chunk_gated_delta_rule + Ascend custom ops: O(T * Nv * Dk * Dv)
    nv = cfg.linear_num_value_heads // tp
    dk = cfg.linear_key_head_dim
    dv = cfg.linear_value_head_dim
    flops_gdn_proxy = 32.0 * float(num_tokens * nv * dk * dv)

    elem = 2  # bf16
    bytes_w = 0
    for p in layer.parameters():
        if p.device == device or p.device.type == device.type:
            bytes_w += p.numel() * elem
    # Activations: inputs + main intermediates + output (same order as forward)
    bytes_activations = num_tokens * hidden * elem * 2  # in + out
    bytes_activations += num_tokens * (out_qkvz // tp + out_ba) * elem
    bytes_activations += num_tokens * nv * dv * elem * 2  # core_attn / z-shaped

    return RooflineEstimates(
        flops_gemm=float(flops_gemm),
        flops_gdn_proxy=float(flops_gdn_proxy),
        bytes_weights=bytes_w,
        bytes_activations=int(bytes_activations),
    )


def _init_dist_hccl(local_rank: int) -> None:
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp}",
        local_rank=local_rank,
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def _load_mapped_weights(model_path: str) -> Iterable[tuple[str, torch.Tensor]]:
    from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {model_path}")

    mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
    for name, tensor in safetensors_weights_iterator(files, use_tqdm_on_load=False):
        yield from mapper.apply([(name, tensor)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
        help="Local HF model directory (same as profile_qwen35_prefill.py)",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU index (see npu-smi info)")
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Decoder layer index whose linear_attn weights to load",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1x4096", "4x4096", "8x2048"],
        metavar="BxT",
        help="Batch x sequence length scenarios (prefill tokens = B*T)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--hidden-source",
        choices=["randn", "file"],
        default="randn",
        help="randn: normal scaled like HF init; file: load .pt tensor [T, H]",
    )
    parser.add_argument("--hidden-file", type=str, default=None, help="Path to tensor for --hidden-source file")
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
        help="Keep default vLLM logging; otherwise set VLLM_LOGGING_LEVEL=ERROR",
    )
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    import torch_npu

    torch_npu.npu.set_device(args.device)
    import vllm_ascend.vllm_ascend_C  # noqa: F401 — register torch.ops._C_ascend (causal conv, etc.)
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()

    # vLLM engine config (no model load — only config objects)
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        _init_dist_hccl(local_rank=args.device)

        # vllm-ascend worker patches (Ascend forward + GDN metadata patches)
        import vllm_ascend.patch.worker  # noqa: F401

        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device("npu", args.device)
        hf_config = vllm_config.model_config.hf_text_config

        # Build full text tower to reuse Qwen3_5Model.load_weights (stacked GDN mapping).
        # Prefix must match vLLM's Qwen3 path so make_layers uses `language_model.model.layers`,
        # not a broken `.layers` when prefix is empty.
        text_model = Qwen3_5Model(vllm_config=vllm_config, prefix="language_model.model")
        prefix_strip = "language_model.model."
        layer_prefix = f"layers.{args.layer}.linear_attn"
        weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in _load_mapped_weights(args.model):
            if not name.startswith(prefix_strip):
                continue
            rest = name[len(prefix_strip) :]
            if not rest.startswith(f"layers.{args.layer}.linear_attn"):
                continue
            weights.append((rest, tensor))

        if not weights:
            raise RuntimeError(
                f"No weights for {prefix_strip}{layer_prefix} under {args.model}. "
                "Check --layer and model path."
            )

        text_model.load_weights(iter(weights))
        layer = text_model.layers[args.layer].linear_attn
        layer = layer.to(device=device, dtype=dtype)

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

        shapes = _parse_shapes(args.shapes)
        print(
            f"Device: npu:{args.device} | layer={args.layer} prefix={layer.prefix!r} | "
            f"warmup={args.warmup} repeats={args.repeats}",
            flush=True,
        )

        for batch, seq_len in shapes:
            num_tokens = batch * seq_len
            query_lens = torch.tensor([seq_len] * batch, dtype=torch.int32, device="cpu")
            q_cpu = torch.zeros(batch + 1, dtype=torch.int32, device="cpu")
            q_cpu[1:] = torch.cumsum(query_lens, dim=0)
            q = q_cpu.to(device=device, non_blocking=True)
            # First prefill: seq_lens == query_lens (no prefix cache)
            seq_lens = query_lens.to(device=device, non_blocking=True)
            block_table = torch.arange(batch, dtype=torch.int32, device=device).unsqueeze(1)

            from vllm.v1.attention.backend import CommonAttentionMetadata

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

            # Hidden states: match llm.generate prefill — post-RMSNorm stream to linear_attn
            hidden_size = hf_config.hidden_size
            if args.hidden_source == "file":
                if not args.hidden_file:
                    raise ValueError("--hidden-file required when --hidden-source=file")
                hs = torch.load(args.hidden_file, map_location=device)
                if hs.shape != (num_tokens, hidden_size):
                    raise ValueError(
                        f"Expected hidden [{num_tokens}, {hidden_size}], got {tuple(hs.shape)}"
                    )
                hidden_states = hs.to(dtype=dtype)
            else:
                # Realistic: same scale as HF init_range (order of magnitude for stable RMSNorm inputs)
                hidden_states = torch.randn(
                    num_tokens, hidden_size, device=device, dtype=dtype
                ) * float(hf_config.initializer_range)

            out_buf = torch.empty(num_tokens, hidden_size, device=device, dtype=dtype)

            # Warmup
            for _ in range(args.warmup):
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    layer.forward(hidden_states, out_buf)
            torch.npu.synchronize()

            times_ms: list[float] = []
            for _ in range(args.repeats):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                    layer.forward(hidden_states, out_buf)
                torch.npu.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

            lat = statistics.median(times_ms) / 1000.0
            est = _estimate_roofline(layer, num_tokens, hidden_size, hf_config, device)

            tflops = est.flops_total / lat / 1e12
            gib_s = est.bytes_total / lat / (1024**3)

            print(
                f"shape batch×seq={batch}×{seq_len} (T={num_tokens}) | "
                f"median {lat*1000:.3f} ms (over {args.repeats} runs) | "
                f"~{tflops:.2f} TFLOP/s (GEMM+proxy) | ~{gib_s:.2f} GiB/s (model bytes)",
                flush=True,
            )

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
