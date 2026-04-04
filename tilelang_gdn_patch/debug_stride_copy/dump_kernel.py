#!/usr/bin/env python3
"""
Dump generated AscendC for ``linear_attention_ker`` (``[B, L, H, D]`` layout).

Uses the workspace ``tilelang-ascend`` tree on ``sys.path`` (not the system
install). Copy ``patched_copy.py`` over ``tilelang/language/copy.py`` in that
tree (or the active install) so BLHD ``T.copy`` regions lower correctly; see README.

Usage:
  python dump_kernel.py
  python dump_kernel.py --tilelang-ascend /path/to/tilelang-ascend
  python dump_kernel.py --out linear_attn_blhd.ascendc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This file: .../<anything>/tilelang_gdn_patch/debug_stride_copy/dump_kernel.py
# Workspace root is three parents up from this directory.
_DEBUG_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _DEBUG_DIR.parents[2]


def _prepend_tilelang_ascend(explicit: str | None) -> Path:
    root = Path(explicit).resolve() if explicit else _WORKSPACE_ROOT / "tilelang-ascend"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def generate_kernel_source(
    h: int,
    d: int,
    c: int,
    *,
    tilelang_ascend: str | Path | None = None,
) -> str:
    """Compile the dump kernel and return AscendC source text.

    Parameters
    ----------
    h, d, c
        Compile-time ``H``, ``D``, and chunk size ``C`` (same as CLI ``--H/--D/--C``).
    tilelang_ascend
        Root of the ``tilelang-ascend`` repo. Default: ``<workspace>/tilelang-ascend``.
    """
    tl_root = _prepend_tilelang_ascend(str(tilelang_ascend) if tilelang_ascend else None)

    if not (tl_root / "tilelang").is_dir():
        raise FileNotFoundError(
            f"expected a tilelang-ascend repo with tilelang/ at {tl_root}"
        )

    if str(_DEBUG_DIR) not in sys.path:
        sys.path.insert(0, str(_DEBUG_DIR))

    from linear_attention_kernel_dump import linear_attention_ker  # noqa: E402

    ker = linear_attention_ker(h, d, c)
    return ker.get_kernel_source()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tilelang-ascend",
        type=str,
        default="",
        help="Root of the tilelang-ascend repo (default: <workspace>/tilelang-ascend)",
    )
    p.add_argument("--H", type=int, default=2, help="num heads (compile-time)")
    p.add_argument("--D", type=int, default=128, help="head dim (compile-time)")
    p.add_argument("--C", type=int, default=64, help="chunk size along L (compile-time)")
    p.add_argument("--out", type=str, default="", help="write AscendC to this file")
    args = p.parse_args()

    try:
        src = generate_kernel_source(
            args.H,
            args.D,
            args.C,
            tilelang_ascend=args.tilelang_ascend or None,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        Path(args.out).write_text(src, encoding="utf-8")
        print(f"Wrote {len(src)} chars to {args.out}", file=sys.stderr)
    else:
        print(src)


if __name__ == "__main__":
    main()
