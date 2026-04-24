#!/usr/bin/env python3
"""Offline vLLM-Ascend prefill-only profiling for Qwen3.5 with torch_npu profiler.

PTO profile trace (JIT chunk kernels instead of Triton GDN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Requires ``vllm_ascend`` installed with the worker hook that reads
``VLLM_PTO_PATCH_DIR`` (see ``vllm_ascend/patch/worker/__init__.py``). The flag
``--pto`` sets that directory before workers spawn so PTO replaces
``chunk_gated_delta_rule`` during prefill (six JIT stages). ``--pto-mega`` sets
``VLLM_PTO_MEGAKERNEL=1`` for the single fused megakernel path; the Chrome trace
should contain the ``PTO_gdn_mega_kernel`` profiler scope.

Example — pick a free NPU, write traces under ``./pto_prefill_trace``::

    export ASCEND_RT_VISIBLE_DEVICES=4
    python3 /workdir/npu_inference_profiling/qwen35_prefill/profile_qwen35_prefill.py \
        --pto \
        --profile-dir /workdir/npu_inference_profiling/qwen35_prefill/pto_prefill_trace \
        --batch-size 1 --seq-len 4096 --max-tokens 1

Equivalent without ``--pto`` (set env before Python starts so spawn inherits it)::

    export ASCEND_RT_VISIBLE_DEVICES=4
    export VLLM_PTO_PATCH_DIR=/workdir/npu_inference_profiling/patch_vllm_pto
    python3 /workdir/npu_inference_profiling/qwen35_prefill/profile_qwen35_prefill.py \
        --profile-dir /workdir/npu_inference_profiling/qwen35_prefill/pto_prefill_trace \
        --batch-size 1 --seq-len 4096 --max-tokens 1

Profiler output: under ``--profile-dir``, the script prints the path to the latest
Chrome trace, typically
``<profile-dir>/rank0_<pid>_..._ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json``.
Engine logs should show that the PTO patch is active when ``--pto`` / env is set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Default model (local snapshot); override with --model.
model_path = "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/"
# model_path = "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Must run before vLLM spawns workers so ``apply_pto_patch`` loads in child processes.
_pto_patch_dir = Path(__file__).resolve().parent.parent / "patch_vllm_pto"
if "--pto-mega" in sys.argv:
    os.environ.setdefault("VLLM_PTO_PATCH_DIR", str(_pto_patch_dir))
    os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
elif "--pto" in sys.argv or os.environ.get("VLLM_USE_PTO_CHUNK") == "1":
    os.environ.setdefault("VLLM_PTO_PATCH_DIR", str(_pto_patch_dir))
    os.environ.pop("VLLM_PTO_MEGAKERNEL", None)

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
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eager mode (default): no CUDA graph / compilation. "
        "Use --no-enforce-eager for graph/compiled mode.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Scheduler token budget per engine step. If unset, set to at least "
        "batch_size * seq_len (+ decode budget) so prefill is one batched pass "
        "(default vLLM ~8192 otherwise causes many small steps and ~batch× kernel launches).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Max concurrent sequences. If unset, uses max(batch_size, 256).",
    )
    parser.add_argument(
        "--pto",
        action="store_true",
        help="Enable PTO per-stage JIT kernels (sets VLLM_PTO_PATCH_DIR for worker import hook).",
    )
    parser.add_argument(
        "--pto-mega",
        action="store_true",
        help="Enable PTO fused megakernel (VLLM_PTO_PATCH_DIR + VLLM_PTO_MEGAKERNEL=1).",
    )
    args = parser.parse_args()
    if args.pto_mega and args.pto:
        parser.error("Use only one of --pto and --pto-mega.")
    if args.pto_mega:
        os.environ.setdefault(
            "VLLM_PTO_PATCH_DIR",
            str(Path(__file__).resolve().parent.parent / "patch_vllm_pto"),
        )
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    elif args.pto:
        os.environ.setdefault(
            "VLLM_PTO_PATCH_DIR",
            str(Path(__file__).resolve().parent.parent / "patch_vllm_pto"),
        )
        os.environ.pop("VLLM_PTO_MEGAKERNEL", None)

    profile_root = args.profile_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = _prompt_with_token_count(tokenizer, args.seq_len)
    prompts = [prompt] * args.batch_size

    # Context must cover prompt + generation.
    max_model_len = max(args.seq_len + args.max_tokens + 8, 4096)

    # vLLM defaults (e.g. max_num_batched_tokens=8192) cap how many tokens can be
    # scheduled together. Below that limit, the engine chunks prefill and/or runs
    # fewer sequences per step — kernel Count in op_statistic.csv then scales with
    # batch_size. Size these from the actual prompt batch (see vLLM #2492).
    prefill_tokens = args.batch_size * args.seq_len
    decode_budget = args.batch_size * args.max_tokens
    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens is None:
        max_num_batched_tokens = prefill_tokens + decode_budget + 1024
    max_num_seqs = args.max_num_seqs if args.max_num_seqs is not None else max(args.batch_size, 256)

    print(
        f"Scheduler: max_num_batched_tokens={max_num_batched_tokens}, "
        f"max_num_seqs={max_num_seqs} (prefill_tokens={prefill_tokens}, "
        f"decode_budget={decode_budget})",
        flush=True,
    )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        profiler_config=_prefill_profiler_config(profile_root),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Warmup run (Triton/JIT, etc.); keep compile out of the profiled window.
    llm.generate(prompts, sampling_params)

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

    if args.pto_mega:
        raw = tp.read_text(encoding="utf-8", errors="ignore")
        if "PTO_gdn_mega_kernel" not in raw:
            raise RuntimeError(
                "Megakernel profiling: expected ``PTO_gdn_mega_kernel`` in trace_view.json; "
                "check that VLLM_PTO_MEGAKERNEL reached workers and the PTO patch is active."
            )
        print("OK: trace contains ``PTO_gdn_mega_kernel`` (megakernel launch path).")


if __name__ == "__main__":
    main()
