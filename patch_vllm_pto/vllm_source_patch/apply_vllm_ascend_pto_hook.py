#!/usr/bin/env python3
"""
Insert the PTO ``apply_pto_patch`` worker hook into the *installed* vllm-ascend package.

Idempotent: if the hook is already present in the correct place (right after
``patch_v2.patch_triton``, before ``patch_weight_utils``), exits 0 without writing.

Usage:
  python3 apply_vllm_ascend_pto_hook.py
  python3 apply_vllm_ascend_pto_hook.py --dry-run
  python3 apply_vllm_ascend_pto_hook.py --vllm-ascend-root /path/to/site-packages/vllm_ascend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Hook body (must stay in sync with README / .patch file)
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

# Unique sentinel inside the hook (used for placement / duplicate detection)
_SENTINEL = "_pto_dir = _pto_os.environ.get(\"VLLM_PTO_PATCH_DIR\")"


def _default_worker_init() -> Path:
    import vllm_ascend

    root = Path(vllm_ascend.__file__).resolve().parent
    return root / "patch" / "worker" / "__init__.py"


def _hook_correctly_placed(text: str) -> bool:
    if _SENTINEL not in text:
        return False
    w = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    if w == -1:
        return False
    return text.find(_SENTINEL) < w


def _remove_legacy_trailing_hook(text: str) -> str:
    """Remove mistaken hook appended *after* ``patch_deepencoder2`` (older draft used EOF placement)."""
    legacy_mark = "# Optional out-of-tree PTO swap for ``chunk_gated_delta_rule``"
    if legacy_mark not in text:
        return text
    idx = text.find(legacy_mark)
    idx_w = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    if idx_w != -1 and idx < idx_w:
        # "Optional" comment is in the correct early region — do not strip.
        return text
    # Trailing legacy: drop from marker through end of file
    if "apply_pto_patch()" not in text[idx:]:
        return text
    return text[:idx].rstrip() + "\n"


def _insert_hook(text: str) -> str:
    anchor = "    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa\n"
    i = text.find(anchor)
    if i == -1:
        raise RuntimeError("Could not find patch_v2.patch_triton import anchor in worker/__init__.py")

    j = i + len(anchor)
    # Skip following newlines
    while j < len(text) and text[j] == "\n":
        j += 1
    # Expect '# isort: off' soon
    rest = text[j:]
    if not rest.lstrip().startswith("# isort: off"):
        raise RuntimeError(
            "Unexpected content after patch_v2.patch_triton: expected blank line(s) then '# isort: off'. "
            "File layout may differ from supported upstream; patch manually (see README.md)."
        )
    insert_at = text.find("# isort: off", j)
    return text[:insert_at] + _HOOK + "\n" + text[insert_at:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vllm-ascend-root",
        type=Path,
        default=None,
        help="Root of vllm_ascend package (directory containing patch/). Default: resolved from import.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions only; do not write.")
    args = ap.parse_args()

    target = (
        args.vllm_ascend_root / "patch" / "worker" / "__init__.py"
        if args.vllm_ascend_root is not None
        else _default_worker_init()
    )
    if not target.is_file():
        print(f"ERROR: not a file: {target}", file=sys.stderr)
        return 2

    original = target.read_text(encoding="utf-8")
    text = _remove_legacy_trailing_hook(original)

    if _hook_correctly_placed(text):
        print(f"OK: PTO hook already present (correct location): {target}")
        return 0

    if _SENTINEL in text:
        print(
            f"ERROR: hook signature found but not before patch_weight_utils: {target}\n"
            "Manual merge required (see vllm_source_patch/README.md).",
            file=sys.stderr,
        )
        return 3

    new_text = _insert_hook(text)
    if new_text == original and not _hook_correctly_placed(new_text):
        print("ERROR: insert failed", file=sys.stderr)
        return 4

    if args.dry_run:
        print(f"DRY-RUN: would write hook into {target}")
        return 0

    target.write_text(new_text, encoding="utf-8")
    print(f"OK: wrote PTO worker hook into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
