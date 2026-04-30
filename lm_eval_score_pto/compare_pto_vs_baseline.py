#!/usr/bin/env python3
"""Compare lm-eval JSON runs: PTO suite vs baseline ``lm_eval_score/outputs/subset_eval/...``.

Reports per-task metric deltas (MMLU accuracies, Wikitext PPL / bpb) and timing
(``eval_execution_seconds``, vLLM tqdm speed fields). Warns if all metrics match
bitwise within tolerance (PTO patch may not be active).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Numeric result leaves: (task, metric_key) e.g. ("mmlu_astronomy", "acc,none")
ResultKey = tuple[str, str]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_result_metrics(results: dict[str, Any]) -> dict[ResultKey, float]:
    out: dict[ResultKey, float] = {}
    for task, payload in results.items():
        if not isinstance(payload, dict):
            continue
        for mk, mv in payload.items():
            if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                out[(task, mk)] = float(mv)
    return out


def _fmt_pct(delta: float, base: float) -> str:
    if base == 0.0:
        return "n/a"
    return f"{100.0 * delta / base:+.4f}%"


def main() -> int:
    root = Path(__file__).resolve().parent
    default_base = root.parent / "lm_eval_score" / "outputs" / "subset_eval" / "run_20260430_153437"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline-root",
        type=Path,
        default=default_base,
        help="Directory containing per-preset subdirs with eval.json",
    )
    ap.add_argument(
        "--pto-root",
        type=Path,
        required=True,
        help="PTO run directory (same layout as baseline)",
    )
    ap.add_argument(
        "--identity-atol",
        type=float,
        default=1e-12,
        help="If all compared metrics within this abs tol, warn about possible inactive patch",
    )
    args = ap.parse_args()

    base_root = args.baseline_root.expanduser().resolve()
    pto_root = args.pto_root.expanduser().resolve()

    if not base_root.is_dir():
        print(f"baseline root not found: {base_root}", file=sys.stderr)
        return 2
    if not pto_root.is_dir():
        print(f"pto root not found: {pto_root}", file=sys.stderr)
        return 2

    presets_base = sorted(p.name for p in base_root.iterdir() if p.is_dir())
    presets_pto = sorted(p.name for p in pto_root.iterdir() if p.is_dir())
    shared = sorted(set(presets_base) & set(presets_pto))
    if not shared:
        print("No shared preset subdirectories between baseline and PTO roots.", file=sys.stderr)
        return 2

    had_metric_pairs = False
    had_metric_diff = False
    timing_lines: list[str] = []

    print(f"baseline: {base_root}")
    print(f"pto:      {pto_root}")
    print()

    for preset in shared:
        jb = base_root / preset / "eval.json"
        jp = pto_root / preset / "eval.json"
        if not jb.is_file() or not jp.is_file():
            print(f"[{preset}] skip — missing eval.json in one side")
            continue

        db, dp = _load(jb), _load(jp)
        rb = db.get("results") or {}
        rp = dp.get("results") or {}
        if not isinstance(rb, dict) or not isinstance(rp, dict):
            print(f"[{preset}] skip — invalid results")
            continue

        mb = _collect_result_metrics(rb)
        mp = _collect_result_metrics(rp)
        keys = sorted(set(mb) & set(mp))

        print(f"=== {preset} ===")
        tb = db.get("timing") or {}
        tp = dp.get("timing") or {}
        if isinstance(tb, dict) and isinstance(tp, dict):
            ev_b = tb.get("eval_execution_seconds")
            ev_p = tp.get("eval_execution_seconds")
            if isinstance(ev_b, (int, float)) and isinstance(ev_p, (int, float)):
                ev_b, ev_p = float(ev_b), float(ev_p)
                delta_ev = ev_p - ev_b
                pct = _fmt_pct(delta_ev, ev_b)
                faster = ""
                if ev_p < ev_b:
                    faster = f" (~{-100.0 * delta_ev / ev_b:.2f}% wall vs baseline eval_execution_seconds)"
                print(
                    f"timing.eval_execution_seconds: baseline={ev_b:.4f} pto={ev_p:.4f} "
                    f"delta={delta_ev:+.4f}s ({pct}){faster}"
                )
                timing_lines.append(f"{preset}\t{ev_b:.4f}\t{ev_p:.4f}\t{delta_ev:+.4f}")
            for fld in (
                "eval_est_speed_input_toks_per_s_last",
                "eval_est_speed_output_toks_per_s_last",
                "eval_est_speed_input_toks_per_s_peak",
                "eval_est_speed_output_toks_per_s_peak",
            ):
                vb, vp = tb.get(fld), tp.get(fld)
                if vb is not None and vp is not None:
                    print(f"  {fld}: {vb} → {vp}")

        for key in keys:
            had_metric_pairs = True
            vb, vp = mb[key], mp[key]
            diff = vp - vb
            same = math.isclose(vp, vb, rel_tol=0.0, abs_tol=args.identity_atol)
            if not same:
                had_metric_diff = True
            task, mk = key
            pct_s = _fmt_pct(diff, vb) if vb != 0 else ""
            print(f"  results[{task}][{mk}]: base={vb:.12g} pto={vp:.12g} Δ={diff:+.6g} ({pct_s})")
        if not keys:
            print("  (no overlapping numeric result keys)")
        print()

    if had_metric_pairs and not had_metric_diff:
        print(
            "WARNING: All compared numeric metrics match baseline within "
            f"identity-atol={args.identity_atol:g}. Double-check that the PTO patch "
            "(coordinator + worker hook + megakernel env) is actually active.",
            file=sys.stderr,
        )

    print("Summary tab-separated (preset, eval_exec_base_s, eval_exec_pto_s, delta_s):")
    for line in timing_lines:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
