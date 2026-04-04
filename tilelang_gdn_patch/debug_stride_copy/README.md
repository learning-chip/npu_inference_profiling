# Strided `T.copy` debugging (AscendC)

AscendC dumps use **`dump_kernel.py`** and **`linear_attention_kernel_dump.py`** only (no imports from other trees for the dump path).

## Patch for `tilelang` (`copy.py`)

This directory ships the full patched module as **`patched_copy.py`** (same path as `tilelang/language/copy.py` inside a `tilelang-ascend` tree).

**Apply** by overwriting the **system- or venv-installed** `copy.py` with this file (back up the original first):

```bash
# Resolve installed path and copy (adjust python if you use a venv)
PY=python3
DST="$($PY -c 'import pathlib, tilelang; print(pathlib.Path(tilelang.__file__).parent / "language" / "copy.py")')"
cp -a "$DST" "${DST}.bak"
cp /path/to/debug_stride_copy/patched_copy.py "$DST"
```

Re-run your tests / workload, then optionally confirm the install matches the shipped file:

```bash
VERIFY_TILELANG_COPY_INSTALLED=1 pytest test_patched_copy.py -v -k installed_copy
```

## Dump generated AscendC

`dump_kernel.py` prepends **`<workspace>/tilelang-ascend`** to `sys.path` by default (workspace = three parents above this folder). Override with **`--tilelang-ascend`** if your clone lives elsewhere. It does **not** modify any installed package.

```bash
python dump_kernel.py --out /tmp/linear_attn_blhd.ascendc
python dump_kernel.py --tilelang-ascend /path/to/tilelang-ascend --out /tmp/out.ascendc
```

Requires a `tilelang-ascend` tree that matches your TVM / CANN runtime (same as normal development).

## Correctness tests (NPU)

**`test_linear_attention_correctness.py`** compares the local kernel (**`linear_attention_kernel_dump`**) to **`ref_linear_attention`** in **`linear_attention_runtime.py`**. All imports stay inside this folder. Requires Ascend NPU; skips when `torch.npu` is missing.

```bash
cd debug_stride_copy
pytest test_patched_copy.py test_linear_attention_correctness.py -v
```

`test_patched_copy.py` checks **`patched_copy.py`** (syntax + BLHD fix markers). With **`VERIFY_TILELANG_COPY_INSTALLED=1`**, it also checks the installed `tilelang.language.copy` file matches **`patched_copy.py`** after you copy it in.

## Root cause (fixed in `patched_copy.py`)

For `T.copy(Q[bz, i * C, by, 0], q_l1)` with `[B, L, H, D]`, region extents must be **`[1, C, 1, D]`**, not **`[1, 1, C, D]`** (which matches `[B, H, L, D]` indexing). Wrong extents produced bad `realTailM` arguments to `copy_gm_to_l1` in generated AscendC (`tl_templates/ascend/common.h`).

**Fix:** `copy.py` adds `_tile_extents_for_buffer_load` to map a 2D tile onto four buffer indices when the L-chunk is on index 1 (BLHD) vs index 2 (BHLD).

## Bisheng / printf (optional)

Use your `tilelang-ascend` tree’s `examples/gemm_aot` flow to compile dumped sources offline.
