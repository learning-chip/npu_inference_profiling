#!/usr/bin/env python3
"""
Compare Triton vs PTO ``chunk_gated_delta_rule`` integration on Qwen3.5 prefill+decode:

- Greedy token IDs for the first generated token and the following ``N-1`` tokens
  (default ``N=11`` → first + next 10).
- Full-vocabulary **log-probabilities** for the **first** decode step (vLLM's
  ``SamplingParams(logprobs=-1)`` / ``CompletionOutput.logprobs[0]``). These are
  ``log_softmax`` scores, i.e. logits minus a row-wise constant; comparing them
  between backends is the right strict check when raw logits are not returned.

Requires a large ``LLM(..., max_logprobs=…)`` (default 300000) so full-vocab
logprobs are not clipped by the engine (default cap is 20).

Each backend runs in a separate subprocess so patches do not cross-contaminate.

**Backend guard:** the child forces ``vllm_ascend.utils.adapt_patch`` then checks
``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`` for the marker
``_vllm_pto_chunk_wrapper_installed`` (set only by ``apply_pto_patch`` / ``bind_triton``).
The Triton baseline uses an environment with **all** ``VLLM_PTO*`` keys and
``VLLM_USE_PTO_CHUNK`` stripped so a parent shell cannot accidentally enable PTO
for both runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_PATCH = Path(__file__).resolve().parent


def _child_env_for_backend(with_pto: bool, artifact: str, base_env: dict[str, str]) -> dict[str, str]:
    """Build subprocess env: baseline drops every ``VLLM_PTO*`` key so we never compare two Triton runs."""
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


def _verify_chunk_backend() -> None:
    """Ensure ``fla.ops.chunk_gated_delta_rule`` matches this subprocess role (PTO wrapper vs Triton-only)."""
    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm_ascend.utils as vua

    vua.adapt_patch(is_global_patch=False)
    want = os.environ.get("_CMP_BACKEND", "").lower().strip()
    fn = fla_ops.chunk_gated_delta_rule
    wrapped = bool(getattr(fn, "_vllm_pto_chunk_wrapper_installed", False))
    if want == "pto" and not wrapped:
        raise RuntimeError(
            "PTO subprocess: expected ``apply_pto_patch`` wrapper on "
            "`vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` "
            f"(missing _vllm_pto_chunk_wrapper_installed; got {fn!r}). "
            "Check VLLM_PTO_PATCH_DIR and vllm_ascend worker import order."
        )
    if want == "triton" and wrapped:
        raise RuntimeError(
            "Triton baseline subprocess: PTO wrapper is still installed on "
            "`fla.ops.chunk_gated_delta_rule` — baseline env was contaminated "
            "(e.g. VLLM_PTO_PATCH_DIR leaked). Strip PTO env vars for the baseline run."
        )
    if want not in ("pto", "triton"):
        raise RuntimeError(f"internal: invalid _CMP_BACKEND={want!r}")


def _dense_first_step_logprobs(
    logprobs_pos0: dict[int, object], vocab_size: int
) -> tuple[np.ndarray, int]:
    """Map vLLM's ``logprobs[0]`` dict to a dense ``float32`` vector ``[vocab_size]``."""
    arr = np.full(vocab_size, -np.inf, dtype=np.float32)
    n_set = 0
    for tid, lp in logprobs_pos0.items():
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size:
            arr[tid_i] = float(getattr(lp, "logprob", lp))
            n_set += 1
    return arr, n_set


def _child_payload() -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # After vLLM import path exists; ensures ``adapt_patch`` sees the same stack as ``LLM()``.
    _verify_chunk_backend()

    model = os.environ["_CMP_MODEL"]
    artifact = os.environ["_CMP_ARTIFACT"]
    n_gen = int(os.environ["_CMP_N_GEN"])
    seed = int(os.environ.get("_CMP_SEED", "42"))

    torch.npu.set_device("npu:0")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    slen = int(os.environ["_CMP_SEQ"])
    seed_text = "The quick brown fox jumps over the lazy dog. "
    ids: list[int] = []
    while len(ids) < slen:
        ids.extend(tok.encode(seed_text, add_special_tokens=False))
    ids = ids[:slen]
    prompt = tok.decode(ids)
    prompts = [prompt]

    # vLLM caps ``logprobs=-1`` at ``EngineArgs.max_logprobs`` (default 20). Qwen3.x vocabs are ~150k–250k.
    max_lp = int(os.environ.get("_CMP_MAX_LOGPROBS", "300000"))

    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(slen + n_gen + 32, 4096),
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=slen + n_gen + 128,
        max_num_seqs=8,
        max_logprobs=max_lp,
    )
    vocab_size = llm.model_config.get_vocab_size()

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=n_gen,
        min_tokens=n_gen,
        ignore_eos=True,
        logprobs=-1,
        seed=seed,
    )
    out = llm.generate(prompts, sp)[0]
    comp = out.outputs[0]
    token_ids = np.asarray(list(comp.token_ids), dtype=np.int32)

    if comp.logprobs is None or len(comp.logprobs) < 1:
        raise RuntimeError("Expected per-step logprobs; got None or empty.")
    lp0_raw = comp.logprobs[0]
    if hasattr(lp0_raw, "keys"):
        lp0 = dict(lp0_raw.items())
    else:
        lp0 = dict(lp0_raw)

    first_lp, n_keys = _dense_first_step_logprobs(lp0, vocab_size)
    if n_keys < vocab_size:
        # Still comparable if both runs omit the same ids; warn for visibility.
        print(
            json.dumps(
                {
                    "warn": f"first-step logprobs dict has {n_keys}/{vocab_size} entries",
                }
            ),
            flush=True,
        )

    np.savez_compressed(
        artifact,
        token_ids=token_ids,
        first_token_logprobs=first_lp,
        vocab_size=np.int32(vocab_size),
    )
    print(
        json.dumps(
            {
                "artifact": artifact,
                "token_ids": token_ids.tolist(),
                "vocab_size": vocab_size,
                "n_logprob_keys": n_keys,
            }
        ),
        flush=True,
    )


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default="4", help="ASCEND_RT_VISIBLE_DEVICES (single NPU index)")
    p.add_argument(
        "--num-generated",
        type=int,
        default=11,
        help="Total greedy tokens to generate (first + next 10 when set to 11).",
    )
    p.add_argument("--seed", type=int, default=42, help="SamplingParams seed (greedy is deterministic).")
    p.add_argument(
        "--logprob-atol",
        type=float,
        default=5e-3,
        help="``allclose`` atol on first-step log-probability vectors.",
    )
    p.add_argument(
        "--logprob-rtol",
        type=float,
        default=2e-2,
        help="``allclose`` rtol on first-step log-probability vectors.",
    )
    p.add_argument(
        "--max-logprobs",
        type=int,
        default=300_000,
        help="Engine ``max_logprobs`` so ``logprobs=-1`` (full vocab) is allowed.",
    )
    args = p.parse_args()

    if args.num_generated < 2:
        p.error("--num-generated must be at least 2 (first token + at least one more).")

    env_base: dict[str, str] = dict(os.environ)
    env_base["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env_base["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    env_base["_CMP_MODEL"] = args.model
    env_base["_CMP_SEQ"] = str(args.seq_len)
    env_base["_CMP_N_GEN"] = str(args.num_generated)
    env_base["_CMP_SEED"] = str(args.seed)
    env_base["_CMP_MAX_LOGPROBS"] = str(args.max_logprobs)

    script = str(Path(__file__).resolve())

    def _run(with_pto: bool) -> dict[str, object]:
        fd, path = tempfile.mkstemp(prefix="pto_cmp_", suffix=".npz")
        os.close(fd)
        try:
            env = _child_env_for_backend(with_pto, path, env_base)
            r = subprocess.run(
                [sys.executable, script, "--internal-child"],
                env=env,
                capture_output=True,
                text=True,
                timeout=1200,
            )
            if r.returncode != 0:
                print(r.stderr[-12000:])
                raise RuntimeError(f"subprocess failed rc={r.returncode}")
            lines = [x for x in r.stdout.splitlines() if x.strip().startswith("{")]
            if not lines:
                print(r.stdout)
                raise RuntimeError("no JSON line in child stdout")
            meta = json.loads(lines[-1])
            z = np.load(meta["artifact"])
            return {
                "meta": meta,
                "token_ids": np.asarray(z["token_ids"]),
                "first_lp": np.asarray(z["first_token_logprobs"]),
                "vocab_size": int(z["vocab_size"]),
            }
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    print("Running Triton baseline subprocess…", flush=True)
    tri = _run(False)
    print("Running PTO-patched subprocess…", flush=True)
    pto = _run(True)

    ids_tri = tri["token_ids"]
    ids_pto = pto["token_ids"]
    print(f"Triton token_ids:    {ids_tri.tolist()}")
    print(f"PTO    token_ids:    {ids_pto.tolist()}")

    if ids_tri.shape != ids_pto.shape or np.any(ids_tri != ids_pto):
        print("MISMATCH: greedy token id sequence differs.")
        return 1
    print("OK: all greedy token ids match.")

    lp_tri = tri["first_lp"]
    lp_pto = pto["first_lp"]
    if lp_tri.shape != lp_pto.shape:
        print("MISMATCH: first-step logprob vector length differs.")
        return 1

    stats = _compare_logprob_vectors(lp_tri, lp_pto)
    print(
        f"First-step log-probs: max_abs={stats['max_abs']:.6g}  "
        f"rmse={stats['rmse']:.6g}  finite={int(stats['n_finite'])} / {lp_tri.size}",
        flush=True,
    )
    close = np.allclose(lp_tri, lp_pto, rtol=args.logprob_rtol, atol=args.logprob_atol, equal_nan=False)
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


if __name__ == "__main__":
    if "--internal-child" in sys.argv:
        _child_payload()
    else:
        raise SystemExit(main())
