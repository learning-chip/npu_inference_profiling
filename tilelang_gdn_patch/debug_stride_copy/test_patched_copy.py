"""
Verify ``patched_copy.py`` (full ``tilelang/language/copy.py`` replacement).

- Always: syntax + required BLHD extent logic is present.
- Optional: after you copy ``patched_copy.py`` over the installed ``copy.py``,
  set ``VERIFY_TILELANG_COPY_INSTALLED=1`` to assert the installed file matches
  ``patched_copy.py`` byte-for-byte.
"""
from __future__ import annotations

import os
import py_compile
from pathlib import Path

import pytest

_DEBUG_DIR = Path(__file__).resolve().parent
_PATCHED = _DEBUG_DIR / "patched_copy.py"


def test_patched_copy_py_exists() -> None:
    assert _PATCHED.is_file(), f"missing {_PATCHED}"


def test_patched_copy_py_compiles() -> None:
    py_compile.compile(str(_PATCHED), doraise=True)


def test_patched_copy_contains_blhd_extent_fix() -> None:
    text = _PATCHED.read_text(encoding="utf-8")
    assert "def _tile_extents_for_buffer_load(" in text
    assert "isinstance(indices[1], tir.Mul)" in text
    assert "return [1, e0, 1, e1]" in text
    assert "mapped = _tile_extents_for_buffer_load(indices, extents)" in text


def test_installed_copy_matches_patched_when_verifying() -> None:
    if os.environ.get("VERIFY_TILELANG_COPY_INSTALLED") != "1":
        pytest.skip("set VERIFY_TILELANG_COPY_INSTALLED=1 after copying patched_copy.py to site-packages")

    import tilelang.language.copy as installed  # noqa: E402

    installed_path = Path(installed.__file__).resolve()
    assert installed_path.read_bytes() == _PATCHED.read_bytes(), (
        f"installed copy.py at {installed_path} differs from {_PATCHED}; "
        "re-copy patched_copy.py to tilelang/language/copy.py"
    )
