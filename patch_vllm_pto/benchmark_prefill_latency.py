#!/usr/bin/env python3
"""TTFT (time-to-first-token) prefill-oriented latency for Qwen3.5 (vLLM-Ascend):
Triton vs PTO stages vs PTO megakernel.

Run each ``--case`` in a **fresh** interpreter so env and JIT state stay isolated::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 benchmark_prefill_latency.py --case triton --seq-len 4096
    python3 benchmark_prefill_latency.py --case pto --seq-len 4096
    python3 benchmark_prefill_latency.py --case pto_mega --seq-len 4096

Emits one JSON object per invocation on stdout. Use ``--output-jsonl PATH`` to
append the same line to a file (newline-delimited JSON) for comparing runs.

Latency is **TTFT** (time to
first token), taken from ``RequestOutput.metrics.first_token_latency`` seconds
— the same notion as ``mean_ttft_ms`` in ``vllm/benchmarks/serve.py`` (prefill +
queue to first generated token), not end-to-end ``generate()`` wall time.

``mean_ms`` / ``std_ms`` / ``times_ms`` duplicate the TTFT stats for backward
compatibility with simple parsers.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
    p.add_argument(
        "--output-jsonl",
        metavar="PATH",
        default=None,
        help="Append this run's JSON object as one line (JSONL) for cross-run comparison.",
    )
    args = p.parse_args()

    _apply_case_env(args.case, args.device)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

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
        disable_log_stats=False,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
        seed=42,
    )

    def _ttft_s(output: RequestOutput) -> float:
        m = output.metrics
        if m is None:
            raise RuntimeError(
                "RequestOutput.metrics is None; TTFT requires disable_log_stats=False "
                "(the default) on LLM()."
            )
        ttft = getattr(m, "first_token_latency", None)
        if ttft is None:
            raise RuntimeError(
                "metrics.first_token_latency missing; use a vLLM build that "
                "populates RequestStateStats on RequestOutput (v1 engine)."
            )
        return float(ttft)

    for _ in range(args.warmup):
        llm.generate(prompts, sp, use_tqdm=False)

    ttfts_s: list[float] = []
    for _ in range(args.repeats):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        if len(outs) != 1:
            raise RuntimeError(f"expected one RequestOutput, got {len(outs)}")
        ttfts_s.append(_ttft_s(outs[0]))

    ttfts_ms = [t * 1000.0 for t in ttfts_s]
    mean_ttft_ms = statistics.mean(ttfts_ms)
    median_ttft_ms = statistics.median(ttfts_ms)
    std_ttft_ms = statistics.pstdev(ttfts_ms) if len(ttfts_ms) > 1 else 0.0
    out = {
        "case": args.case,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ttft_ms": mean_ttft_ms,
        "median_ttft_ms": median_ttft_ms,
        "std_ttft_ms": std_ttft_ms,
        "ttfts_ms": ttfts_ms,
        "mean_ms": mean_ttft_ms,
        "std_ms": std_ttft_ms,
        "times_ms": ttfts_ms,
    }
    line = json.dumps(out)
    print(line, flush=True)
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
