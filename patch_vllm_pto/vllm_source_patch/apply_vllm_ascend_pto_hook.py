#!/usr/bin/env python3
"""
Apply vLLM-Ascend in-tree edits needed for out-of-tree PTO ``chunk_gated_delta_rule``:

1. **worker/__init__.py** — optional ``VLLM_PTO_PATCH_DIR`` hook (early, inside ``HAS_TRITON``).
2. **patch_qwen3_5.py** and **patch_qwen3_next.py** — use ``_vllm_fla_ops.chunk_gated_delta_rule``
   at **call time** so monkey-patches on ``vllm.model_executor.layers.fla.ops`` take effect even if
   ``apply_pto_patch()`` runs after those modules were first imported.

Idempotent for each step.

Usage:
  python3 apply_vllm_ascend_pto_hook.py
  python3 apply_vllm_ascend_pto_hook.py --dry-run
  python3 apply_vllm_ascend_pto_hook.py --vllm-ascend-root /path/to/site-packages/vllm_ascend
  python3 apply_vllm_ascend_pto_hook.py --skip-qwen-dynamic-import
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Hook body (must stay in sync with README / patches/worker__init__pto_hook.patch)
_HOOK = '''
    # Out-of-tree PTO swap: MUST run before ``patch_qwen3_next`` / ``patch_qwen3_5`` (and any
    # patch that does ``from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule``),
    # or those modules keep a stale reference to Triton and never call the patched op.
    try:
        import os as _pto_os
        import sys as _pto_sys

        _pto_dir = _pto_os.environ.get("VLLM_PTO_PATCH_DIR")
        if _pto_dir and _pto_os.path.isdir(_pto_dir):
            if _pto_dir not in _pto_sys.path:
                _pto_sys.path.insert(0, _pto_dir)
            from apply import apply_pto_patch  # type: ignore  # noqa: E402

            apply_pto_patch()
    except Exception as _pto_exc:
        import warnings as _pto_warnings

        _pto_warnings.warn(f"VLLM_PTO_PATCH_DIR apply_pto_patch failed: {_pto_exc!r}", stacklevel=1)

'''

_SENTINEL = "_pto_dir = _pto_os.environ.get(\"VLLM_PTO_PATCH_DIR\")"


def _default_vllm_ascend_root() -> Path:
    import vllm_ascend

    return Path(vllm_ascend.__file__).resolve().parent


def _worker_init(root: Path) -> Path:
    return root / "patch" / "worker" / "__init__.py"


def _hook_correctly_placed(text: str) -> bool:
    if _SENTINEL not in text:
        return False
    w = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    if w == -1:
        return False
    return text.find(_SENTINEL) < w


def _remove_legacy_trailing_hook(text: str) -> str:
    legacy_mark = "# Optional out-of-tree PTO swap for ``chunk_gated_delta_rule``"
    if legacy_mark not in text:
        return text
    idx = text.find(legacy_mark)
    idx_w = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    if idx_w != -1 and idx < idx_w:
        return text
    if "apply_pto_patch()" not in text[idx:]:
        return text
    return text[:idx].rstrip() + "\n"


def _insert_worker_hook(text: str) -> str:
    anchor = "    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa\n"
    i = text.find(anchor)
    if i == -1:
        raise RuntimeError("Could not find patch_v2.patch_triton import anchor in worker/__init__.py")

    j = i + len(anchor)
    while j < len(text) and text[j] == "\n":
        j += 1
    rest = text[j:]
    if not rest.lstrip().startswith("# isort: off"):
        raise RuntimeError(
            "Unexpected content after patch_v2.patch_triton: expected blank line(s) then '# isort: off'."
        )
    insert_at = text.find("# isort: off", j)
    return text[:insert_at] + _HOOK + "\n" + text[insert_at:]


def _qwen_dynamic_import_ok(text: str) -> bool:
    return "_vllm_fla_ops.chunk_gated_delta_rule" in text


def _patch_qwen3_5(path: Path) -> str | None:
    """Return new text if changed, else None."""
    t = path.read_text(encoding="utf-8")
    if _qwen_dynamic_import_ok(t):
        return None
    old = "from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule"
    new = (
        "import vllm.model_executor.layers.fla.ops as _vllm_fla_ops\n"
        "from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule"
    )
    if old not in t:
        raise RuntimeError(f"{path}: expected import line not found; merge manually.")
    t = t.replace(old, new, 1)
    t = t.replace(") = chunk_gated_delta_rule(\n", ") = _vllm_fla_ops.chunk_gated_delta_rule(\n", 1)
    return t


def _patch_qwen3_next(path: Path) -> str | None:
    t = path.read_text(encoding="utf-8")
    if _qwen_dynamic_import_ok(t):
        return None
    old = "from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule\n"
    new = "import vllm.model_executor.layers.fla.ops as _vllm_fla_ops\n"
    if old not in t:
        raise RuntimeError(f"{path}: expected import line not found; merge manually.")
    t = t.replace(old, new, 1)
    t = t.replace(") = chunk_gated_delta_rule(\n", ") = _vllm_fla_ops.chunk_gated_delta_rule(\n", 1)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vllm-ascend-root",
        type=Path,
        default=None,
        help="Root of vllm_ascend package. Default: resolved from ``import vllm_ascend``.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions only; do not write.")
    ap.add_argument(
        "--skip-worker-hook",
        action="store_true",
        help="Do not modify patch/worker/__init__.py.",
    )
    ap.add_argument(
        "--skip-qwen-dynamic-import",
        action="store_true",
        help="Do not modify patch_qwen3_5.py / patch_qwen3_next.py.",
    )
    args = ap.parse_args()

    root = args.vllm_ascend_root if args.vllm_ascend_root is not None else _default_vllm_ascend_root()

    if not args.skip_worker_hook:
        target = _worker_init(root)
        if not target.is_file():
            print(f"ERROR: not a file: {target}", file=sys.stderr)
            return 2
        original = target.read_text(encoding="utf-8")
        text = _remove_legacy_trailing_hook(original)
        if _hook_correctly_placed(text):
            print(f"OK: PTO worker hook already present: {target}")
        elif _SENTINEL in text:
            print(
                f"ERROR: worker hook found but misplaced: {target}",
                file=sys.stderr,
            )
            return 3
        else:
            new_text = _insert_worker_hook(text)
            if args.dry_run:
                print(f"DRY-RUN: would write worker hook -> {target}")
            else:
                target.write_text(new_text, encoding="utf-8")
                print(f"OK: wrote worker hook -> {target}")

    if not args.skip_qwen_dynamic_import:
        for rel, fn in (
            ("patch_qwen3_5.py", _patch_qwen3_5),
            ("patch_qwen3_next.py", _patch_qwen3_next),
        ):
            p = root / "patch" / "worker" / rel
            if not p.is_file():
                print(f"ERROR: missing {p}", file=sys.stderr)
                return 4
            new_t = fn(p)
            if new_t is None:
                print(f"OK: Qwen FLA dynamic import already applied: {p}")
            elif args.dry_run:
                print(f"DRY-RUN: would patch {p}")
            else:
                p.write_text(new_t, encoding="utf-8")
                print(f"OK: patched {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
