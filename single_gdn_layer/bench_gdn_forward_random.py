#!/usr/bin/env python3
"""
NPU benchmark for AscendQwen3_5GatedDeltaNet.forward with *random* weights (no checkpoints).

Uses the same vLLM/V1 GDN prefill path as ``bench_gdn_forward.py``, but builds
``Qwen3_5GatedDeltaNet`` directly and initializes parameters with ``torch.nn.init`` —
no ``safetensors``, no weight iterators, no ``Qwen3_5Model.load_weights``.

Architecture shapes are fixed in this file for:
  - **0.8B** — Qwen3.5-0.8B-class ``text_config`` (hidden 1024, 24 layers pattern, 16/16 GDN heads)
  - **9B** — Qwen3.5-9B-class ``text_config`` (hidden 4096, 32 layers pattern, 16/32 GDN heads)

A minimal ``config.json`` is written to a temporary directory only so ``EngineArgs``
can build ``VllmConfig`` (same as the real benchmark). No model weights are read.

Compare against ``bench_gdn_forward.py`` (real weights) on the same ``--shapes``;
median latency should match within run-to-run variance.

**Profiling:** pass ``--profile`` to capture an Ascend ``torch_npu`` trace (same family of
API as ``qwen35_prefill/profile_qwen35_prefill.py`` / vLLM worker: CPU+NPU activities,
tensorboard_trace_handler output under ``--profile-dir``, then a ``.zip`` of that tree).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# ---------------------------------------------------------------------------
# Qwen3.5 text/vision shapes (HF Qwen3_5ForConditionalGeneration), no weight files.
# layer_types: same pattern as official configs (full_attention every 4th layer).
# ---------------------------------------------------------------------------
def _qwen35_layer_types(num_hidden_layers: int, full_every: int = 4) -> list[str]:
    return [
        "full_attention" if (i + 1) % full_every == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    ]


def _build_qwen35_hf_config(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    linear_num_value_heads: int,
    vision_depth: int,
    vision_hidden_size: int,
    vision_intermediate_size: int,
    vision_num_heads: int,
    vision_out_hidden_size: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": intermediate_size,
            "layer_types": _qwen35_layer_types(num_hidden_layers),
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": linear_num_value_heads,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": tie_word_embeddings,
            "use_cache": True,
            "vocab_size": 248320,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
        "tie_word_embeddings": tie_word_embeddings,
        "transformers_version": "4.57.0.dev0",
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": vision_depth,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": vision_hidden_size,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": vision_intermediate_size,
            "model_type": "qwen3_5",
            "num_heads": vision_num_heads,
            "num_position_embeddings": 2304,
            "out_hidden_size": vision_out_hidden_size,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


# Qwen3.5-0.8B and Qwen3.5-9B (dense text) — dimensions from official config.json.
QWEN35_0_8B_CONFIG = _build_qwen35_hf_config(
    hidden_size=1024,
    intermediate_size=3584,
    num_hidden_layers=24,
    num_attention_heads=8,
    num_key_value_heads=2,
    linear_num_value_heads=16,
    vision_depth=12,
    vision_hidden_size=768,
    vision_intermediate_size=3072,
    vision_num_heads=12,
    vision_out_hidden_size=1024,
    tie_word_embeddings=True,
)

QWEN35_9B_CONFIG = _build_qwen35_hf_config(
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    linear_num_value_heads=32,
    vision_depth=27,
    vision_hidden_size=1152,
    vision_intermediate_size=4304,
    vision_num_heads=16,
    vision_out_hidden_size=4096,
    tie_word_embeddings=False,
)


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
    tp = layer.tp_size
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    out_qkvz = 2 * key_dim + 2 * value_dim
    flops_qkvz = 2 * num_tokens * hidden * (out_qkvz // tp)
    nh = cfg.linear_num_value_heads
    out_ba = 2 * (nh // tp)
    flops_ba = 2 * num_tokens * hidden * out_ba
    flops_out = 2 * num_tokens * (value_dim // tp) * hidden
    conv_dim = key_dim * 2 + value_dim
    k = cfg.linear_conv_kernel_dim
    flops_conv = 2 * num_tokens * conv_dim * k
    flops_gemm = flops_qkvz + flops_ba + flops_out + flops_conv
    nv = cfg.linear_num_value_heads // tp
    dk = cfg.linear_key_head_dim
    dv = cfg.linear_value_head_dim
    flops_gdn_proxy = 32.0 * float(num_tokens * nv * dk * dv)
    elem = 2
    bytes_w = sum(p.numel() * elem for p in layer.parameters() if p.device.type == device.type)
    bytes_activations = num_tokens * hidden * elem * 2
    bytes_activations += num_tokens * (out_qkvz // tp + out_ba) * elem
    bytes_activations += num_tokens * nv * dv * elem * 2
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


def _write_config_temp(cfg: dict[str, Any]) -> str:
    d = tempfile.mkdtemp(prefix="qwen35_cfg_")
    path = os.path.join(d, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return d


def _randomize_gdn_layer(layer: torch.nn.Module, initializer_range: float, seed: int) -> None:
    """Normal init on all parameters (same std as HF ``initializer_range``)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=initializer_range)


def _create_ascend_torch_profiler(
    profile_dir: Path,
    worker_name: str,
    *,
    with_stack: bool,
    with_memory: bool = False,
):
    """Ascend ``torch_npu.profiler`` setup aligned with ``vllm_ascend.worker.worker._create_profiler``."""
    import torch_npu

    profile_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=with_memory,
        with_modules=with_stack,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(profile_dir.resolve()),
            worker_name=worker_name,
        ),
    )


def _find_latest_ascend_chrome_trace(profile_dir: Path) -> Path | None:
    """Same layout as ``qwen35_prefill/profile_qwen35_prefill.py`` (MindStudio / chrome://tracing)."""
    trace_paths = sorted(
        profile_dir.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return trace_paths[0] if trace_paths else None


def _zip_profile_tree(profile_dir: Path) -> Path:
    """Zip ``profile_dir`` for download; returns path to the ``.zip`` file."""
    profile_dir = profile_dir.resolve()
    if not profile_dir.is_dir():
        raise FileNotFoundError(profile_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_base = profile_dir.parent / f"{profile_dir.name}_{stamp}"
    archive_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(profile_dir))
    return Path(archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["0.8B", "9B"],
        default="0.8B",
        help="Which inlined Qwen3.5 text/vision shape set to use",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU index")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (for prefix string only)")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1x4096", "4x4096", "8x2048"],
        metavar="BxT",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
        help="If set, do not force VLLM_LOGGING_LEVEL=ERROR",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="After timing, capture Ascend torch_npu profiler trace (see --profile-dir).",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "gdn_random_torch_profile",
        help="Directory for torch_npu tensorboard_trace_handler output (created if missing).",
    )
    parser.add_argument(
        "--profile-worker-name",
        type=str,
        default="gdn_random_forward",
        help="Worker name passed to tensorboard_trace_handler (trace subfolder name).",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Enable with_modules (Python stacks); slower, same idea as vLLM torch_profiler_with_stack.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable profiler memory tracking (torch_npu profile_memory).",
    )
    parser.add_argument(
        "--profile-shapes",
        choices=["first", "all"],
        default="first",
        help="Which --shapes entries to include in the profiled region (default: first only).",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=3,
        help="Warmup forwards immediately before starting the profiler (JIT/Triton).",
    )
    parser.add_argument(
        "--profile-skip-zip",
        action="store_true",
        help="Do not create a .zip of --profile-dir after profiling.",
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

    engine_args = EngineArgs(
        model=tmp_model_dir,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    vllm_config = engine_args.create_engine_config()

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        _init_dist_hccl(local_rank=args.device)

        import vllm_ascend.patch.worker  # noqa: F401

        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

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

        shapes = _parse_shapes(args.shapes)
        hidden_size = hf_text.hidden_size

        def forward_shape(batch: int, seq_len: int) -> None:
            """One GDN prefill forward (random activations deterministic in ``seed``)."""
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

            torch.manual_seed(args.seed + num_tokens)
            hidden_states = (
                torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
                * float(hf_text.initializer_range)
            )

            out_buf = torch.empty(num_tokens, hidden_size, device=device, dtype=dtype)

            with set_forward_context(attn_dict, vllm_config, num_tokens=num_tokens):
                layer.forward(hidden_states, out_buf)

        print(
            f"[random weights] variant={args.variant} | npu:{args.device} | "
            f"prefix={layer.prefix!r} | seed={args.seed} | warmup={args.warmup} repeats={args.repeats}",
            flush=True,
        )

        for batch, seq_len in shapes:
            num_tokens = batch * seq_len

            for _ in range(args.warmup):
                forward_shape(batch, seq_len)
            torch.npu.synchronize()

            times_ms: list[float] = []
            for _ in range(args.repeats):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                forward_shape(batch, seq_len)
                torch.npu.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

            lat = statistics.median(times_ms) / 1000.0
            est = _estimate_roofline(layer, num_tokens, hidden_size, hf_text, device)
            tflops = est.flops_total / lat / 1e12
            gib_s = est.bytes_total / lat / (1024**3)

            print(
                f"shape batch×seq={batch}×{seq_len} (T={num_tokens}) | "
                f"median {lat*1000:.3f} ms | ~{tflops:.2f} TFLOP/s (GEMM+proxy) | ~{gib_s:.2f} GiB/s",
                flush=True,
            )

        if args.profile:
            profile_shapes = shapes if args.profile_shapes == "all" else shapes[:1]
            prof_dir = args.profile_dir
            print(
                f"[profile] Ascend torch_npu profiler → {prof_dir.resolve()} | "
                f"shapes={profile_shapes} | worker={args.profile_worker_name!r}",
                flush=True,
            )
            # Clean KV for a deterministic trace (timing runs left cache dirty).
            conv_state.zero_()
            ssm_state.zero_()

            for _ in range(args.profile_warmup):
                for b, t in profile_shapes:
                    forward_shape(b, t)
                torch.npu.synchronize()

            profiler = _create_ascend_torch_profiler(
                prof_dir,
                args.profile_worker_name,
                with_stack=args.profile_with_stack,
                with_memory=args.profile_memory,
            )
            profiler.start()
            try:
                for b, t in profile_shapes:
                    forward_shape(b, t)
                    torch.npu.synchronize()
            finally:
                torch.npu.synchronize()
                profiler.stop()
            torch.npu.synchronize()
            # Allow Ascend profiler worker to flush (see profile_qwen35_prefill.py).
            time.sleep(5)

            trace = _find_latest_ascend_chrome_trace(prof_dir)
            if trace is None:
                print(
                    f"[profile] WARNING: no trace_view.json under {prof_dir}/*_ascend_pt/... "
                    "Profiling may have failed; check NPU logs.",
                    flush=True,
                )
            else:
                sz = trace.stat().st_size
                print(
                    f"[profile] Chrome trace (MindStudio Insight / chrome://tracing): {trace} ({sz} bytes)",
                    flush=True,
                )

            if not args.profile_skip_zip:
                zpath = _zip_profile_tree(prof_dir)
                print(f"[profile] Zipped archive for download: {zpath}", flush=True)

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
