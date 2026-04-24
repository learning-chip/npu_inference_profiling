#!/usr/bin/env python3
"""Wall-clock prefill latency for Qwen3.5 (vLLM-Ascend): Triton vs PTO stages vs PTO megakernel.

Run each ``--case`` in a **fresh** interpreter so env and JIT state stay isolated::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 benchmark_prefill_latency.py --case triton --seq-len 4096
    python3 benchmark_prefill_latency.py --case pto --seq-len 4096
    python3 benchmark_prefill_latency.py --case pto_mega --seq-len 4096

Emits one JSON object per invocation (mean_ms, std_ms, …) on stdout.

Timing is wall-clock around blocking ``LLM.generate()`` (vLLM runs NPU work in
worker processes; driver-side ``torch.npu.synchronize()`` does not wait on them).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

_PATCH = Path(__file__).resolve().parent


def _apply_case_env(case: str, device: str) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
    for k in list(os.environ.keys()):
        if k.startswith("VLLM_PTO"):
            del os.environ[k]
    os.environ.pop("VLLM_USE_PTO_CHUNK", None)
    if case == "triton":
        pass
    elif case == "pto":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
    elif case == "pto_mega":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    else:
        raise ValueError(case)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("triton", "pto", "pto_mega"), required=True)
    p.add_argument(
        "--model",
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    )
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--device", default="0", help="ASCEND_RT_VISIBLE_DEVICES (single NPU index)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1, help="Keep small so timing is prefill-dominated.")
    args = p.parse_args()

    _apply_case_env(args.case, args.device)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = "The quick brown fox jumps over the lazy dog. "
    ids: list[int] = []
    while len(ids) < args.seq_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    ids = ids[: args.seq_len]
    prompt = tok.decode(ids)
    prompts = [prompt]

    max_model_len = max(args.seq_len + args.max_tokens + 32, 4096)
    prefill_tokens = args.seq_len + args.max_tokens + 256

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=prefill_tokens,
        max_num_seqs=8,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
        seed=42,
    )

    for _ in range(args.warmup):
        llm.generate(prompts, sp)

    times_s: list[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        times_s.append(time.perf_counter() - t0)

    mean_ms = statistics.mean(times_s) * 1000.0
    std_ms = statistics.pstdev(times_s) * 1000.0 if len(times_s) > 1 else 0.0
    out = {
        "case": args.case,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "times_ms": [t * 1000.0 for t in times_s],
    }
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
