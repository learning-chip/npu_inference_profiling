#!/usr/bin/env python3
"""
Compare Triton vs PTO ``chunk_gated_delta_rule`` on Qwen3.5 prefill+decode:

- Greedy token IDs for the first generated token and the following ``N-1`` tokens
  (default ``N=11`` → first + next 10).
- Full-vocabulary **log-probabilities** for the **first** decode step (vLLM's
  ``SamplingParams(logprobs=-1)``). Use ``LLM(..., max_logprobs=…)`` (default
  300000) so ``logprobs=-1`` is not clipped (engine default cap is 20).

**No nested Python subprocess:** vLLM already spawns engine workers. Run each
backend in a **separate process** by invoking this script twice (``record``),
then ``compare`` on the two ``.npz`` files — or use ``run_compare_prefill.sh``.

**Backend guard:** ``record`` sets ``_CMP_BACKEND`` and either strips all
``VLLM_PTO*`` (Triton) or sets ``VLLM_PTO_PATCH_DIR`` (PTO), then imports vLLM
and checks ``fla.ops.chunk_gated_delta_rule`` for
``_vllm_pto_chunk_wrapper_installed``.

Examples::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 compare_prefill_next_token.py record --backend triton --output ./tmp/tri.npz
    python3 compare_prefill_next_token.py record --backend pto --output ./tmp/pto.npz
    python3 compare_prefill_next_token.py compare ./tmp/tri.npz ./tmp/pto.npz

Or::

    ./run_compare_prefill.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PATCH = Path(__file__).resolve().parent


def _strip_pto_from_environ() -> None:
    for k in list(os.environ.keys()):
        if k.startswith("VLLM_PTO"):
            del os.environ[k]
    os.environ.pop("VLLM_USE_PTO_CHUNK", None)


def _child_env_for_backend(with_pto: bool, artifact: str, base_env: dict[str, str]) -> dict[str, str]:
    """Build env dict for **external** launchers (e.g. Bash) — not used by this file's record/compare."""
    env: dict[str, str] = {}
    for k, v in base_env.items():
        if k.startswith("VLLM_PTO"):
            continue
        env[k] = v
    env.pop("VLLM_USE_PTO_CHUNK", None)
    env["_CMP_ARTIFACT"] = artifact
    if with_pto:
        env["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
        env["_CMP_BACKEND"] = "pto"
    else:
        env.pop("VLLM_PTO_PATCH_DIR", None)
        env["_CMP_BACKEND"] = "triton"
    return env


def _apply_record_environ(*, backend: str, device: str) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
    os.environ["_CMP_BACKEND"] = backend
    if backend == "pto":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
    elif backend == "triton":
        _strip_pto_from_environ()
    else:
        raise ValueError(f"backend must be triton or pto, got {backend!r}")


def _verify_chunk_backend() -> None:
    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm_ascend.utils as vua

    vua.adapt_patch(is_global_patch=False)
    want = os.environ.get("_CMP_BACKEND", "").lower().strip()
    fn = fla_ops.chunk_gated_delta_rule
    wrapped = bool(getattr(fn, "_vllm_pto_chunk_wrapper_installed", False))
    if want == "pto" and not wrapped:
        raise RuntimeError(
            "PTO record: expected ``apply_pto_patch`` wrapper on "
            "`vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` "
            f"(missing _vllm_pto_chunk_wrapper_installed; got {fn!r}). "
            "Check VLLM_PTO_PATCH_DIR and vllm_ascend worker import order."
        )
    if want == "triton" and wrapped:
        raise RuntimeError(
            "Triton baseline record: PTO wrapper is still installed on "
            "`fla.ops.chunk_gated_delta_rule` — strip VLLM_PTO* from the environment."
        )
    if want not in ("pto", "triton"):
        raise RuntimeError(f"internal: invalid _CMP_BACKEND={want!r}")


def _dense_first_step_logprobs(
    logprobs_pos0: dict[int, object], vocab_size: int
) -> tuple[np.ndarray, int]:
    arr = np.full(vocab_size, -np.inf, dtype=np.float32)
    n_set = 0
    for tid, lp in logprobs_pos0.items():
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size:
            arr[tid_i] = float(getattr(lp, "logprob", lp))
            n_set += 1
    return arr, n_set


def cmd_record(args: argparse.Namespace) -> int:
    """Load vLLM once in this process with the chosen backend; write ``.npz``."""
    _apply_record_environ(backend=args.backend, device=args.device)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    _verify_chunk_backend()

    torch.npu.set_device("npu:0")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed_text = "The quick brown fox jumps over the lazy dog. "
    ids: list[int] = []
    while len(ids) < args.seq_len:
        ids.extend(tok.encode(seed_text, add_special_tokens=False))
    ids = ids[: args.seq_len]
    prompt = tok.decode(ids)
    prompts = [prompt]

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(args.seq_len + args.num_generated + 32, 4096),
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=args.seq_len + args.num_generated + 128,
        max_num_seqs=8,
        max_logprobs=args.max_logprobs,
    )
    vocab_size = llm.model_config.get_vocab_size()

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.num_generated,
        min_tokens=args.num_generated,
        ignore_eos=True,
        logprobs=-1,
        seed=args.seed,
    )
    out = llm.generate(prompts, sp)[0]
    comp = out.outputs[0]
    token_ids = np.asarray(list(comp.token_ids), dtype=np.int32)

    if comp.logprobs is None or len(comp.logprobs) < 1:
        raise RuntimeError("Expected per-step logprobs; got None or empty.")
    lp0_raw = comp.logprobs[0]
    lp0 = dict(lp0_raw.items()) if hasattr(lp0_raw, "keys") else dict(lp0_raw)

    first_lp, n_keys = _dense_first_step_logprobs(lp0, vocab_size)
    if n_keys < vocab_size:
        print(
            json.dumps({"warn": f"first-step logprobs dict has {n_keys}/{vocab_size} entries"}),
            flush=True,
        )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        token_ids=token_ids,
        first_token_logprobs=first_lp,
        vocab_size=np.int32(vocab_size),
    )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "output": str(outp),
                "token_ids": token_ids.tolist(),
                "vocab_size": vocab_size,
                "n_logprob_keys": n_keys,
            }
        ),
        flush=True,
    )
    return 0


def _compare_logprob_vectors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return {"max_abs": float("nan"), "rmse": float("nan"), "n_finite": 0.0}
    d_f = d[finite]
    return {
        "max_abs": float(np.max(np.abs(d_f))),
        "rmse": float(np.sqrt(np.mean(d_f.astype(np.float64) ** 2))),
        "n_finite": float(np.sum(finite)),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    tri_path, pto_path = Path(args.triton_npz), Path(args.pto_npz)
    zt = np.load(tri_path)
    zp = np.load(pto_path)
    ids_tri = np.asarray(zt["token_ids"])
    ids_pto = np.asarray(zp["token_ids"])
    print(f"Triton token_ids:    {ids_tri.tolist()}")
    print(f"PTO    token_ids:    {ids_pto.tolist()}")

    if ids_tri.shape != ids_pto.shape or np.any(ids_tri != ids_pto):
        print("MISMATCH: greedy token id sequence differs.")
        return 1
    print("OK: all greedy token ids match.")

    lp_tri = np.asarray(zt["first_token_logprobs"])
    lp_pto = np.asarray(zp["first_token_logprobs"])
    if lp_tri.shape != lp_pto.shape:
        print("MISMATCH: first-step logprob vector length differs.")
        return 1

    stats = _compare_logprob_vectors(lp_tri, lp_pto)
    print(
        f"First-step log-probs: max_abs={stats['max_abs']:.6g}  "
        f"rmse={stats['rmse']:.6g}  finite={int(stats['n_finite'])} / {lp_tri.size}",
        flush=True,
    )
    close = np.allclose(
        lp_tri, lp_pto, rtol=args.logprob_rtol, atol=args.logprob_atol, equal_nan=False
    )
    if not close:
        worst = int(np.nanargmax(np.abs(lp_tri - lp_pto)))
        print(
            f"MISMATCH: first-step log-probs not within atol={args.logprob_atol} "
            f"rtol={args.logprob_rtol} (worst token id={worst}).",
            flush=True,
        )
        return 1
    print(
        f"OK: first-step full-vocab log-probs match (atol={args.logprob_atol}, rtol={args.logprob_rtol}).",
        flush=True,
    )
    return 0


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default="0", help="ASCEND_RT_VISIBLE_DEVICES (single NPU index)")
    p.add_argument("--num-generated", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-logprobs", type=int, default=300_000)


def main() -> int:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("record", help="Run one backend and save .npz (start a fresh Python process per backend).")
    _add_common_model_args(p_rec)
    p_rec.add_argument("--backend", choices=("triton", "pto"), required=True)
    p_rec.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p_rec.set_defaults(func=cmd_record)

    p_cmp = sub.add_parser("compare", help="Compare two .npz files from record (no vLLM import).")
    p_cmp.add_argument("triton_npz", type=str)
    p_cmp.add_argument("pto_npz", type=str)
    p_cmp.add_argument("--logprob-atol", type=float, default=5e-3)
    p_cmp.add_argument("--logprob-rtol", type=float, default=2e-2)
    p_cmp.set_defaults(func=cmd_compare)

    args = root.parse_args()
    if args.command == "record" and args.num_generated < 2:
        root.error("record: --num-generated must be at least 2")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
