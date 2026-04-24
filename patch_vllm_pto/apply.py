"""Monkey-patch vLLM FLA ``chunk_gated_delta_rule`` to use PTO kernels (prefill)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_PATCH_ACTIVE = False
_log = logging.getLogger(__name__)


def _maybe_pto_isa_path() -> None:
    if "PTO_LIB_PATH" in os.environ:
        return
    fb = "/sources/pto-isa"
    if os.path.isdir(os.path.join(fb, "include")):
        os.environ["PTO_LIB_PATH"] = fb


def apply_pto_patch() -> None:
    """Call after ``vllm_ascend`` worker patches (so Triton is the baseline we wrap)."""
    global _PATCH_ACTIVE
    _maybe_pto_isa_path()
    import vllm.model_executor.layers.fla.ops as fla_ops

    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule as triton_chunk_gated_delta_rule

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from pto_chunk_gated_delta_rule import bind_triton

    fla_ops.chunk_gated_delta_rule = bind_triton(triton_chunk_gated_delta_rule)
    _PATCH_ACTIVE = True
    _mega = os.environ.get("VLLM_PTO_MEGAKERNEL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _log.warning(
        "PTO chunk_gated_delta_rule patch is active (fused megakernel, C=128)."
        if _mega
        else "PTO chunk_gated_delta_rule patch is active (6 JIT kernels, C=128).",
    )


def is_pto_patch_active() -> bool:
    return _PATCH_ACTIVE
