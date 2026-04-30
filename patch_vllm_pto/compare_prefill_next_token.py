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
    python3 compare_prefill_next_token.py record --backend pto_mega --output ./tmp/mega.npz
    python3 compare_prefill_next_token.py compare ./tmp/tri.npz ./tmp/pto.npz
    python3 compare_prefill_next_token.py logprob_alignment ./tmp/tri.npz ./tmp/pto.npz ./tmp/mega.npz --out-dir ./figs

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
    os.environ.pop("VLLM_PTO_MEGAKERNEL", None)


def _child_env_for_backend(with_pto: bool, artifact: str, base_env: dict[str, str]) -> dict[str, str]:
    """Build env dict for **external** launchers (e.g. Bash) — not used by this file's record/compare."""
    env: dict[str, str] = {}
    for k, v in base_env.items():
        if k.startswith("VLLM_PTO"):
            continue
        env[k] = v
    env.pop("VLLM_USE_PTO_CHUNK", None)
    env.pop("VLLM_PTO_MEGAKERNEL", None)
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
        os.environ.pop("VLLM_PTO_MEGAKERNEL", None)
    elif backend == "pto_mega":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    elif backend == "triton":
        _strip_pto_from_environ()
    else:
        raise ValueError(f"backend must be triton, pto, or pto_mega, got {backend!r}")


def _apply_pto_patch_driver_early() -> None:
    """Bind PTO wrapper in this process **before** ``import vllm``, not only in workers."""
    pt = os.environ.get("VLLM_PTO_PATCH_DIR")
    if not pt or not os.path.isdir(pt):
        return
    if pt not in sys.path:
        sys.path.insert(0, pt)
    from apply import apply_pto_patch  # noqa: E402

    apply_pto_patch()


def _verify_chunk_backend() -> None:
    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm_ascend.utils as vua

    want = os.environ.get("_CMP_BACKEND", "").lower().strip()
    vua.adapt_patch(is_global_patch=False)
    if want in ("pto", "pto_mega"):
        _apply_pto_patch_driver_early()

    fn = fla_ops.chunk_gated_delta_rule
    wrapped = bool(getattr(fn, "_vllm_pto_chunk_wrapper_installed", False))
    if want in ("pto", "pto_mega") and not wrapped:
        raise RuntimeError(
            "PTO record: expected ``apply_pto_patch`` wrapper on "
            "`vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` "
            f"(missing _vllm_pto_chunk_wrapper_installed; got {fn!r}). "
            "Check VLLM_PTO_PATCH_DIR and vllm_ascend worker import order."
        )
    if want == "pto_mega" and os.environ.get("VLLM_PTO_MEGAKERNEL", "").strip() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise RuntimeError(
            "pto_mega record: expected VLLM_PTO_MEGAKERNEL to enable the fused kernel path."
        )
    if want == "triton" and wrapped:
        raise RuntimeError(
            "Triton baseline record: PTO wrapper is still installed on "
            "`fla.ops.chunk_gated_delta_rule` — strip VLLM_PTO* from the environment."
        )
    if want not in ("pto", "triton", "pto_mega"):
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

    llm_kw: dict = dict(
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
    if getattr(args, "quantization", None):
        llm_kw["quantization"] = args.quantization
    llm = LLM(**llm_kw)
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


def _distribution_from_log_probs(lp: np.ndarray) -> np.ndarray:
    """Normalize exp(log-probabilities) over finite coordinates to a probability vector."""
    finite = np.isfinite(lp)
    out = np.zeros(lp.shape[0], dtype=np.float64)
    if not np.any(finite):
        return out
    lpf = lp[finite].astype(np.float64)
    mx = np.max(lpf)
    e = np.exp(lpf - mx)
    den = np.sum(e)
    if den <= 0.0:
        return out
    out[finite] = e / den
    return out


def _kl_divergence_pq(p_ref: np.ndarray, p_cand: np.ndarray) -> float:
    """D_KL(p_ref ‖ p_cand) in nats (natural log); both must be nonnegative and sum ~1."""
    eps = 1e-300
    p = np.asarray(p_ref, dtype=np.float64)
    q = np.asarray(p_cand, dtype=np.float64)
    return float(np.sum(np.where(p > eps, p * (np.log(p + eps) - np.log(q + eps)), 0.0)))


def _logprob_alignment_metrics(lp_ref: np.ndarray, lp_cand: np.ndarray) -> dict[str, float]:
    """First-step alignment: RMSE / relative RMSE (log-prob space) and KL with Triton distribution as reference."""
    finite = np.isfinite(lp_ref) & np.isfinite(lp_cand)
    if not np.any(finite):
        return {
            "max_abs": float("nan"),
            "rmse": float("nan"),
            "rel_rmse_rms_ref": float("nan"),
            "rel_rmse_std_ref": float("nan"),
            "kl_ref_vs_cand": float("nan"),
            "n_finite": 0.0,
        }
    ref_f = lp_ref[finite].astype(np.float64)
    cand_f = lp_cand[finite].astype(np.float64)
    d = cand_f - ref_f
    rmse = float(np.sqrt(np.mean(d**2)))
    rms_ref = float(np.sqrt(np.mean(ref_f**2)))
    std_ref = float(np.std(ref_f))
    rel_rms = rmse / rms_ref if rms_ref > 1e-30 else float("nan")
    rel_std = rmse / std_ref if std_ref > 1e-30 else float("nan")
    p_ref = _distribution_from_log_probs(lp_ref)
    p_cand = _distribution_from_log_probs(lp_cand)
    kl = _kl_divergence_pq(p_ref, p_cand)
    return {
        "max_abs": float(np.max(np.abs(d))),
        "rmse": rmse,
        "rel_rmse_rms_ref": rel_rms,
        "rel_rmse_std_ref": rel_std,
        "kl_ref_vs_cand": kl,
        "n_finite": float(np.sum(finite)),
    }


def _scatter_logprob_vs_ref(
    lp_ref: np.ndarray,
    lp_cand: np.ndarray,
    out_path: Path,
    cand_label: str,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    finite = np.isfinite(lp_ref) & np.isfinite(lp_cand)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        raise RuntimeError("no finite overlapping log-prob entries for scatter")
    if idx.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)
    x = lp_ref[idx].astype(np.float64)
    y = lp_cand[idx].astype(np.float64)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.02 + 1e-6
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(x, y, s=3, alpha=0.2, c="#1f77b4", edgecolors="none", rasterized=True)
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Triton log-prob (first decode step)")
    ax.set_ylabel(f"{cand_label} log-prob (first decode step)")
    ax.set_title(f"{cand_label} vs Triton (subsample ≤{max_points} tokens)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _print_alignment_block(name: str, m: dict[str, float]) -> None:
    print(
        f"[{name}] first-step log-probs vs Triton (reference), "
        f"n_finite={int(m['n_finite'])}:",
        flush=True,
    )
    print(
        f"  RMSE(cand - ref) = {m['rmse']:.6g}\n"
        f"  relative RMSE = RMSE / RMS(ref) = {m['rel_rmse_rms_ref']:.6g}\n"
        f"  relative RMSE = RMSE / std(ref) = {m['rel_rmse_std_ref']:.6g}\n"
        f"  max_abs = {m['max_abs']:.6g}\n"
        f"  D_KL(Triton || {name}) = {m['kl_ref_vs_cand']:.6g} nats",
        flush=True,
    )


def cmd_logprob_alignment(args: argparse.Namespace) -> int:
    """Scatter + RMSE / relative RMSE / KL for first-step log-probs in ``.npz`` files."""
    tri_path = Path(args.triton_npz)
    pto_path = Path(args.pto_npz)
    zt = np.load(tri_path)
    lp_tri = np.asarray(zt["first_token_logprobs"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_pts = int(args.scatter_max_points)

    zp = np.load(pto_path)
    lp_pto = np.asarray(zp["first_token_logprobs"])
    if lp_tri.shape != lp_pto.shape:
        print("MISMATCH: Triton vs PTO first_token_logprobs length.", flush=True)
        return 1
    m_pto = _logprob_alignment_metrics(lp_tri, lp_pto)
    _print_alignment_block("PTO", m_pto)
    pto_png = out_dir / "scatter_pto_vs_triton.png"
    _scatter_logprob_vs_ref(lp_tri, lp_pto, pto_png, "PTO", max_pts)
    print(f"Wrote {pto_png}", flush=True)

    if args.pto_mega_npz:
        zm = np.load(Path(args.pto_mega_npz))
        lp_m = np.asarray(zm["first_token_logprobs"])
        if lp_tri.shape != lp_m.shape:
            print("MISMATCH: Triton vs PTO-mega first_token_logprobs length.", flush=True)
            return 1
        m_m = _logprob_alignment_metrics(lp_tri, lp_m)
        _print_alignment_block("PTO-mega", m_m)
        mega_png = out_dir / "scatter_pto_mega_vs_triton.png"
        _scatter_logprob_vs_ref(lp_tri, lp_m, mega_png, "PTO-mega", max_pts)
        print(f"Wrote {mega_png}", flush=True)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    tri_path, pto_path = Path(args.triton_npz), Path(args.pto_npz)
    zt = np.load(tri_path)
    zp = np.load(pto_path)
    ids_tri = np.asarray(zt["token_ids"])
    ids_pto = np.asarray(zp["token_ids"])
    print(f"Triton token_ids:    {ids_tri.tolist()}")
    print(f"Candidate token_ids: {ids_pto.tolist()}")

    if ids_tri.shape != ids_pto.shape or np.any(ids_tri != ids_pto):
        print("MISMATCH: greedy token id sequence differs.")
        return 1
    print("OK: all greedy token ids match.")

    lp_tri = np.asarray(zt["first_token_logprobs"])
    lp_pto = np.asarray(zp["first_token_logprobs"])
    if lp_tri.shape != lp_pto.shape:
        print("MISMATCH: first-step logprob vector length differs.")
        return 1

    cand_tag = pto_path.name.replace(".npz", "")
    metrics = _logprob_alignment_metrics(lp_tri, lp_pto)
    _print_alignment_block(cand_tag, metrics)
    scatter_out = getattr(args, "scatter_out", None)
    if scatter_out is not None:
        outp = Path(scatter_out)
        _scatter_logprob_vs_ref(
            lp_tri,
            lp_pto,
            outp,
            cand_label=cand_tag,
            max_points=int(args.scatter_max_points),
        )
        print(f"Wrote scatter plot {outp}", flush=True)

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
    p.add_argument(
        "--quantization",
        default=None,
        metavar="METHOD",
        help="Pass to LLM(.., quantization=…) e.g. ascend for msmodelslim W8A8 weights.",
    )


def main() -> int:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("record", help="Run one backend and save .npz (start a fresh Python process per backend).")
    _add_common_model_args(p_rec)
    p_rec.add_argument(
        "--backend",
        choices=("triton", "pto", "pto_mega"),
        required=True,
        help="``pto`` = six PTO stages; ``pto_mega`` = single fused megakernel launch.",
    )
    p_rec.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p_rec.set_defaults(func=cmd_record)

    p_cmp = sub.add_parser("compare", help="Compare two .npz files from record (no vLLM import).")
    p_cmp.add_argument("triton_npz", type=str)
    p_cmp.add_argument("pto_npz", type=str)
    p_cmp.add_argument("--logprob-atol", type=float, default=5e-3)
    p_cmp.add_argument("--logprob-rtol", type=float, default=2e-2)
    p_cmp.add_argument(
        "--scatter-out",
        type=Path,
        default=None,
        metavar="PATH.png",
        help="Optional 1:1 scatter of first-step log-probs (Triton x-axis, candidate y-axis).",
    )
    p_cmp.add_argument(
        "--scatter-max-points",
        type=int,
        default=10_000,
        metavar="N",
        help="Random subsample cap for scatter plots (full vocab used for metrics).",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_la = sub.add_parser(
        "logprob_alignment",
        help="Print RMSE / relative RMSE / KL vs Triton and write scatter PNG(s) from .npz (no vLLM).",
    )
    p_la.add_argument("triton_npz", type=str)
    p_la.add_argument("pto_npz", type=str)
    p_la.add_argument(
        "pto_mega_npz",
        nargs="?",
        default=None,
        help="Optional third .npz for PTO-mega (second scatter.png + metrics block).",
    )
    p_la.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for scatter_pto_vs_triton.png and scatter_pto_mega_vs_triton.png",
    )
    p_la.add_argument("--scatter-max-points", type=int, default=10_000, metavar="N")
    p_la.set_defaults(func=cmd_logprob_alignment)

    args = root.parse_args()
    if args.command == "record" and args.num_generated < 2:
        root.error("record: --num-generated must be at least 2")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
