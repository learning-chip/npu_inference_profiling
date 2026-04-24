# vLLM-Ascend in-tree changes for PTO GDN

Out-of-tree code under `patch_vllm_pto/` (`apply.py`, `pto_chunk_gated_delta_rule.py`, …) can replace `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` **only if** `apply_pto_patch()` runs in every engine **worker** process **before** any module executes:

```python
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
```

Otherwise those modules bind the **Triton** implementation at import time and keep calling it forever, even if `fla.ops.chunk_gated_delta_rule` is monkey-patched later.

That ordering cannot be enforced from the out-of-tree directory alone: it requires a **small edit** inside the installed `vllm_ascend` package.

## File and change

| Path (inside the `vllm_ascend` package) | Change |
|----------------------------------------|--------|
| `vllm_ascend/patch/worker/__init__.py` | Immediately after `import vllm_ascend.patch.worker.patch_v2.patch_triton` (still inside `if HAS_TRITON:`), run the optional `VLLM_PTO_PATCH_DIR` hook: prepend that directory to `sys.path`, `from apply import apply_pto_patch`, `apply_pto_patch()`. Must appear **before** `import vllm_ascend.patch.worker.patch_qwen3_next` and `patch_qwen3_5`. |

## Environment

- **`VLLM_PTO_PATCH_DIR`**: absolute path to the directory that contains `apply.py` (this repo’s `npu_inference_profiling/patch_vllm_pto/`).

## Legacy mistake (remove if present)

Older drafts appended the same hook **at the end** of `worker/__init__.py`. That runs **after** `patch_qwen3_5` imports `chunk_gated_delta_rule`, so the trace still showed Triton `chunk_gated_delta_rule_fwd`. If you have that trailing block, delete it and apply the hook in the correct place only once.

## Apply on a fresh install

From a machine with official `vllm-ascend` + `vllm` already importable:

```bash
python3 /path/to/npu_inference_profiling/patch_vllm_pto/vllm_source_patch/apply_vllm_ascend_pto_hook.py
```

Or:

```bash
/path/to/npu_inference_profiling/patch_vllm_pto/vllm_source_patch/apply_vllm_ascend_pto_hook.sh
```

The script is **idempotent**: if the hook is already present immediately after `patch_v2.patch_triton`, it exits successfully without edits.

## Reference unified diff

See `patches/worker__init__pto_hook.patch` (paths are relative to the parent of the `vllm_ascend` package directory, e.g. `site-packages/`).

## Revert

Remove the `try:` / `apply_pto_patch()` block between `patch_v2.patch_triton` and `# isort: off` (the block that references `VLLM_PTO_PATCH_DIR`). Restore the original two blank lines if desired. Do not remove `patch_triton` imports.
