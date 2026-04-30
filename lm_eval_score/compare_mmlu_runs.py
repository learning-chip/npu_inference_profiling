#!/usr/bin/env python3
"""Compare two JSON payloads written by ``run_mmlu_vllm.py --output-json``.

Reports whether aggregate ``groups`` / per-task metrics match (within tolerance for floats).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _floatish(x: Any) -> bool:
    return isinstance(x, bool) is False and isinstance(x, (int, float))


def _collect_metrics(prefix: str, obj: Any, out: list[tuple[str, float]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_metrics(f"{prefix}/{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_metrics(f"{prefix}[{i}]", v, out)
    elif _floatish(obj):
        out.append((prefix, float(obj)))


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, float]] = []
    if data.get("groups"):
        _collect_metrics("groups", data["groups"], pairs)
    if data.get("results"):
        _collect_metrics("results", data["results"], pairs)
    return dict(pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()

    ma, mb = load_metrics(args.run_a), load_metrics(args.run_b)
    keys_a, keys_b = set(ma), set(mb)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    if only_a or only_b:
        print("Key mismatch:", file=sys.stderr)
        if only_a:
            print(f"  only in A ({len(only_a)}):", only_a[:20], "...", file=sys.stderr)
        if only_b:
            print(f"  only in B ({len(only_b)}):", only_b[:20], "...", file=sys.stderr)
        return 2

    mismatches: list[tuple[str, float, float]] = []
    for k in sorted(keys_a):
        va, vb = ma[k], mb[k]
        if not math.isclose(va, vb, rel_tol=args.rtol, abs_tol=args.atol):
            mismatches.append((k, va, vb))

    header_mmlu = [k for k in ma if k.startswith("groups/mmlu") and k.endswith("/acc,none")]
    for k in sorted(header_mmlu):
        print(f"{k}: A={ma[k]:.12g} B={mb[k]:.12g} match={math.isclose(ma[k], mb[k], rel_tol=args.rtol, abs_tol=args.atol)}")

    print(f"Compared {len(keys_a)} numeric leaves.")
    if mismatches:
        print(f"MISMATCH count={len(mismatches)} (showing up to 30):")
        for k, va, vb in mismatches[:30]:
            print(f"  {k}: A={va:.12g} B={vb:.12g} diff={vb - va:.3g}")
        return 1

    print("All compared numeric metrics match within rtol/atol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
