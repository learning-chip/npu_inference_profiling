# PTO patch for vLLM-Ascend (Qwen3.5 GDN)

Out-of-tree replacement for FLA **chunk_gated_delta_rule** prefill with PTO JIT kernels, plus small validation scripts.

## Layout

| Path | Role |
|------|------|
| `apply.py` | `apply_pto_patch()` — bind PTO wrapper on `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule` |
| `pto_chunk_gated_delta_rule.py` | PTO forward, `record_function("PTO_gdn_*")` scopes |
| `compare_prefill_next_token.py` | Greedy tokens + first-step full-vocab logprobs: **`record`** / **`compare`** |
| `run_compare_prefill.sh` | Two shell-level `python … record` invocations + `compare` (no nested Python `subprocess`) |
| `compare_pto_triton_lm_eval.py` | Wikitext token PPL + optional MMLU (file-based child results; see script doc) |
| `sanity_pto_triton_eval.sh` | Tiny Bash-driven eval with `timeout(1)` |
| `vllm_source_patch/` | How to patch installed `vllm_ascend` worker imports |

## Prefill parity (`compare_prefill_next_token.py`)

vLLM already spawns engine subprocesses. This repo **does not** nest another Python `subprocess.run` around each backend (avoids `Popen.communicate` stalls when worker logs fill pipes).

**Usage**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
./run_compare_prefill.sh
```

Or manually:

```bash
python3 compare_prefill_next_token.py record --backend triton --output /tmp/tri.npz --device 0
python3 compare_prefill_next_token.py record --backend pto   --output /tmp/pto.npz --device 0
python3 compare_prefill_next_token.py compare /tmp/tri.npz /tmp/pto.npz
```

### Last measured parity (2026-04-24, Ascend 910B, NPU 0)

Command: `OUTDIR=/tmp/prefill_cmp_test3 SEQ_LEN=128 NUM_GEN=11 ./run_compare_prefill.sh`  
Model: `Qwen3.5-0.8B` snapshot `2fc06364715b967f1860aea9cf38778875588b17`  
`max_logprobs=300000`, `SamplingParams(logprobs=-1)` on first decode step.

| Check | Result |
|--------|--------|
| Greedy token IDs (11 tokens) | **Identical** Triton vs PTO: `[15217, 5388, 13, 561, 3841, 13477, 37550, 33075, 888, 279, 15217]` |
| First-step logprob vector | `max_abs = 0.218746`, `rmse = 0.0433051`, `248320 / 248320` finite entries |
| `numpy.allclose` | **PASS** with default `atol=0.005`, `rtol=0.02` |

**Note:** A shorter smoke run (`SEQ_LEN=64`, `NUM_GEN=5`) exceeded the default `allclose` bound on some logits while greedy tokens still matched; use the default **128-token** prompt (or relax tolerances) when probing very short prefills.

## Worker hook

Set `VLLM_PTO_PATCH_DIR` to this directory before starting vLLM so `vllm_ascend` can import `apply_pto_patch()`. See `vllm_source_patch/README.md` for patching an installed `vllm_ascend` tree (worker `__init__.py` + Qwen worker modules).
