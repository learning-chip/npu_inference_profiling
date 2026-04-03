#!/usr/bin/env python3
"""Offline vLLM-Ascend prefill-only profiling for Qwen3.5 with torch_npu profiler."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Default model (local snapshot); override with --model.
model_path = "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/"
# model_path = "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import ProfilerConfig  # noqa: E402


def _prompt_with_token_count(tokenizer: AutoTokenizer, num_tokens: int) -> str:
    """Build a text prompt whose tokenization length is exactly ``num_tokens``."""
    seed = "The quick brown fox jumps over the lazy dog. "
    ids: list[int] = []
    while len(ids) < num_tokens:
        ids.extend(tokenizer.encode(seed, add_special_tokens=False))
    ids = ids[:num_tokens]
    return tokenizer.decode(ids)


def _prefill_profiler_config(profile_dir: Path) -> ProfilerConfig:
    """Match vLLM ``offline_inference/run_one_batch.py`` profile=prefill semantics."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profile_dir.resolve()),
        torch_profiler_with_stack=True,
        delay_iterations=0,
        max_iterations=1,
        ignore_frontend=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=model_path, help="Model path or HF id")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096, help="Prompt length in tokens")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "vllm_prefill_profile_eager",
        help="Absolute or relative directory for torch profiler output",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Decode tokens per request (keep small for prefill-focused runs)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    profile_root = args.profile_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = _prompt_with_token_count(tokenizer, args.seq_len)
    prompts = [prompt] * args.batch_size

    # Context must cover prompt + generation.
    max_model_len = max(args.seq_len + args.max_tokens + 8, 4096)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        profiler_config=_prefill_profiler_config(profile_root),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )

    llm.start_profile()
    llm.generate(prompts, sampling_params)
    llm.stop_profile()

    # Allow worker processes to flush profiler output (see vLLM simple_profiling.py).
    time.sleep(5)

    # Ascend PyTorch Profiler emits a `*_ascend_pt` folder; Chrome trace is trace_view.json.
    trace_paths = sorted(
        profile_root.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not trace_paths:
        raise RuntimeError(
            f"No trace_view.json under {profile_root}/*_ascend_pt/ASCEND_PROFILER_OUTPUT/. "
            "Profiling may have failed; check vLLM / NPU logs."
        )
    tp = trace_paths[0]
    if tp.stat().st_size < 1024:
        raise RuntimeError(f"Trace file looks too small: {tp}")
    print(f"Profiler output directory: {profile_root.resolve()}")
    print(f"Chrome trace (MindStudio Insight / chrome://tracing): {tp} ({tp.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
