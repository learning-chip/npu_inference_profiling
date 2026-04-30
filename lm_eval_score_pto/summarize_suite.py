#!/usr/bin/env python3
"""Emit a FULL_SUITE_REPORT-style table from a subset_eval run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _acc(data: dict[str, Any], task: str) -> str:
    r = data.get("results") or {}
    if not isinstance(r, dict):
        return "n/a"
    t = r.get(task)
    if not isinstance(t, dict):
        return "n/a"
    v = t.get("acc,none")
    return f"{float(v):.4f}" if isinstance(v, (int, float)) else "n/a"


def _wiki(data: dict[str, Any]) -> tuple[str, str]:
    r = data.get("results") or {}
    if not isinstance(r, dict):
        return "n/a", "n/a"
    w = r.get("wikitext")
    if not isinstance(w, dict):
        return "n/a", "n/a"
    wp = w.get("word_perplexity,none")
    bp = w.get("bits_per_byte,none")
    ws = f"{float(wp):.4f}" if isinstance(wp, (int, float)) else "n/a"
    bs = f"{float(bp):.4f}" if isinstance(bp, (int, float)) else "n/a"
    return ws, bs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite_root", type=Path, help="e.g. outputs/subset_eval/run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()
    root = args.suite_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    presets = sorted(p.name for p in root.iterdir() if p.is_dir())
    tasks = (
        "mmlu_astronomy",
        "mmlu_high_school_mathematics",
        "mmlu_college_biology",
        "mmlu_high_school_world_history",
        "mmlu_professional_law",
        "mmlu_philosophy",
    )

    hdr = (
        "preset | max_len | gpu_util | EP | wall_s | astronomy | high_school_mathematics | "
        "college_biology | high_school_world_history | professional_law | philosophy | wiki_w_ppl | wiki_bpb"
    )
    sep = (
        "-------+---------+----------+-----+--------+-----------+-------------------------+"
        "-----------------+---------------------------+------------------+------------+------------+---------"
    )
    lines = [
        f"Suite directory: {root}",
        "",
        hdr,
        sep,
    ]

    for preset in presets:
        jp = root / preset / "eval.json"
        if not jp.is_file():
            continue
        data = _load(jp)
        ma = data.get("model_args") or {}
        if not isinstance(ma, dict):
            ma = {}
        mlen = ma.get("max_model_len", "n/a")
        gu = ma.get("gpu_memory_utilization", "n/a")
        ep = ma.get("enable_expert_parallel", False)
        ep_s = str(bool(ep))
        tm = data.get("timing") or {}
        wall = tm.get("wall_clock_total_seconds") if isinstance(tm, dict) else None
        wall_s = f"{float(wall):.1f}" if isinstance(wall, (int, float)) else "n/a"
        cols = [preset, str(mlen), str(gu), ep_s, wall_s]
        for t in tasks:
            cols.append(_acc(data, t))
        wp, bp = _wiki(data)
        cols.extend([wp, bp])
        lines.append(" | ".join(cols))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
