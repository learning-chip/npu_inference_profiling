#!/usr/bin/env python3
"""TTFT (time-to-first-token) prefill-oriented latency for Qwen3.5 (vLLM-Ascend):
Triton vs PTO stages vs PTO megakernel.

Use a **fresh** interpreter per ``--case`` (and per model) so env and JIT stay
isolated between backends. Pass **multiple** ``--seq-len`` values to load vLLM
once and sweep lengths in-process (saves slow init; see
``run_benchmark_prefill_three_way.sh``)::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 benchmark_prefill_latency.py --case triton --seq-len 512 1024 4096
    python3 benchmark_prefill_latency.py --case pto --seq-len 512 1024 4096

Emits one JSON object per ``--seq-len`` on stdout. Use ``--output-jsonl PATH``
to append each line (JSONL). ``run_benchmark_prefill_three_way.sh`` writes under
``OUT_DIR/<model_label>/`` (e.g. ``…/0.8B/triton.jsonl``) with ``model`` and
``model_label`` fields on each line.

Latency is **TTFT** (time to
first token), taken from ``RequestOutput.metrics.first_token_latency`` seconds
— the same notion as ``mean_ttft_ms`` in ``vllm/benchmarks/serve.py`` (prefill +
queue to first generated token), not end-to-end ``generate()`` wall time.

``mean_ms`` / ``std_ms`` / ``times_ms`` duplicate the TTFT stats for backward
compatibility with simple parsers.

Each JSON line includes ``input_tps``: prompt tokens per second from mean TTFT,
``seq_len / (mean_ttft_ms / 1000)`` (reliable for prefill-heavy single-request
timing).

``vllm_interval_prompt_throughput_tps`` is the engine's
``LoggingStatLogger.last_prompt_throughput`` (``Avg prompt throughput`` in
``vllm/v1/metrics/loggers.py``): a rolling-window value that only updates on
vLLM's stats log interval, so it is often ``null`` or ``0.0`` for short sweeps
and is included only for comparison with engine logs.

Prefix caching (APC) is forced **off** via ``enable_prefix_caching=False`` so
repeated prompts and multi-``seq_len`` sweeps do not reuse KV blocks; TTFT stays
prefill-dominated without APC speedups.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

_PATCH = Path(__file__).resolve().parent


def _infer_model_label(model_path: str) -> str:
    """Best-effort size tag from HF-style path (e.g. ``…Qwen3.5-0.8B…`` → ``0.8B``)."""
    m = re.search(r"Qwen3\.5-(\d+(?:\.\d+)?)B", model_path, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}B"
    m2 = re.search(r"-(\d+(?:\.\d+)?)B(?:/|$)", model_path)
    if m2:
        return f"{m2.group(1)}B"
    return "unknown"


def _vllm_last_logged_prompt_throughput_tps(llm) -> float | None:
    """Read ``LoggingStatLogger.last_prompt_throughput`` for engine 0 (or aggregate).

    Updates only when vLLM runs ``do_log_stats`` / ``_update_stats()`` on its
    timer, so this is often ``0.0`` and is **not** a per-``seq_len`` benchmark
    figure — use ``input_tps`` (from TTFT) for that.
    """
    mgr = getattr(llm.llm_engine, "logger_manager", None)
    if mgr is None:
        return None
    for sl in mgr.stat_loggers:
        per_engine = getattr(sl, "per_engine_stat_loggers", None)
        if isinstance(per_engine, dict) and 0 in per_engine:
            inner = per_engine[0]
            if hasattr(inner, "last_prompt_throughput"):
                return float(inner.last_prompt_throughput)
        if hasattr(sl, "last_prompt_throughput") and not isinstance(
            getattr(sl, "per_engine_stat_loggers", None), dict
        ):
            return float(sl.last_prompt_throughput)
    return None


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
    p.add_argument(
        "--model-label",
        default=None,
        metavar="TAG",
        help="Short label for JSON/JSONL (e.g. 0.8B, 9B). Default: infer from --model path.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[4096],
        metavar="N",
        help="One or more prompt lengths; one LLM load, then sweep in order.",
    )
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
    model_label = (args.model_label or "").strip() or _infer_model_label(args.model)

    _apply_case_env(args.case, args.device)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    seq_lens: list[int] = list(args.seq_len)
    max_sl = max(seq_lens)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = "The quick brown fox jumps over the lazy dog. "

    def _prompt_for_seq_len(seq_len: int) -> list[str]:
        ids: list[int] = []
        while len(ids) < seq_len:
            ids.extend(tok.encode(seed, add_special_tokens=False))
        ids = ids[:seq_len]
        return [tok.decode(ids)]

    max_model_len = max(max_sl + args.max_tokens + 32, 4096)
    prefill_tokens = max_sl + args.max_tokens + 256

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
        enable_prefix_caching=False,
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

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for seq_len in seq_lens:
        prompts = _prompt_for_seq_len(seq_len)

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
        mean_ttft_s = mean_ttft_ms / 1000.0
        input_tps = (seq_len / mean_ttft_s) if mean_ttft_s > 0 else 0.0
        vllm_interval_tps = _vllm_last_logged_prompt_throughput_tps(llm)
        out = {
            "case": args.case,
            "seq_len": seq_len,
            "input_tps": input_tps,
            "median_ttft_ms": median_ttft_ms,
            "mean_ttft_ms": mean_ttft_ms,
            "std_ttft_ms": std_ttft_ms,
            "vllm_interval_prompt_throughput_tps": vllm_interval_tps,
            "model_label": model_label,
            "warmup": args.warmup,
            "repeats": args.repeats,
        }
        line = json.dumps(out)
        print(line, flush=True)
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
