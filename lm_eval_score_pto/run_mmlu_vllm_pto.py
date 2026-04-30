#!/usr/bin/env python3
"""Coordinator wrapper: activate PTO chunk_gated_delta_rule (megakernel default), then run ``lm_eval_score/run_mmlu_vllm.py``.

Workers still require ``VLLM_PTO_PATCH_DIR`` plus an installed ``vllm_ascend`` worker hook (see ``patch_vllm_pto/vllm_source_patch/``).

Extra CLI (stripped before delegation):

    --no-pto-megakernel   Use staged PTO kernels instead of fused megakernel.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PROFILING_ROOT = _ROOT.parent
_PATCH_DIR = (_PROFILING_ROOT / "patch_vllm_pto").resolve()
_DELEGATE = (_PROFILING_ROOT / "lm_eval_score" / "run_mmlu_vllm.py").resolve()


def _strip_wrapper_argv(argv: list[str]) -> tuple[list[str], bool]:
    mega = True
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--no-pto-megakernel":
            mega = False
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out, mega


def _prepare_pto(*, megakernel: bool) -> None:
    if not _PATCH_DIR.is_dir():
        raise RuntimeError(f"VLLM PTO patch dir missing: {_PATCH_DIR}")
    if not _DELEGATE.is_file():
        raise RuntimeError(f"Delegate script missing: {_DELEGATE}")

    os.environ["VLLM_PTO_PATCH_DIR"] = str(_PATCH_DIR)
    if megakernel:
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    else:
        os.environ.pop("VLLM_PTO_MEGAKERNEL", None)

    import vllm  # noqa: F401 — ensure package graph loads before patching

    import vllm_ascend.utils as vua

    vua.adapt_patch(is_global_patch=False)

    pt = str(_PATCH_DIR)
    if pt not in sys.path:
        sys.path.insert(0, pt)
    from apply import apply_pto_patch  # noqa: E402

    apply_pto_patch()

    import vllm.model_executor.layers.fla.ops as fla_ops

    fn = fla_ops.chunk_gated_delta_rule
    if not getattr(fn, "_vllm_pto_chunk_wrapper_installed", False):
        raise RuntimeError(
            "PTO wrapper not installed on chunk_gated_delta_rule "
            "(missing _vllm_pto_chunk_wrapper_installed). "
            "Check VLLM_PTO_PATCH_DIR and vllm_ascend worker hook "
            "(apply_vllm_ascend_pto_hook.py)."
        )


def main() -> None:
    filtered, mega = _strip_wrapper_argv(sys.argv[1:])
    sys.argv = [sys.argv[0]] + filtered
    _prepare_pto(megakernel=mega)
    runpy.run_path(str(_DELEGATE), run_name="__main__")


if __name__ == "__main__":
    main()
