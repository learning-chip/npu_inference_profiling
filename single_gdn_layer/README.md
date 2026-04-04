# Single GDN layer benchmarks

Micro-benchmarks for **`AscendQwen3_5GatedDeltaNet.forward`** (vLLM + vllm-ascend), prefill path only.

| Script | Weights | Notes |
|--------|---------|--------|
| `bench_gdn_forward.py` | Real checkpoint (`*.safetensors`) | Loads one decoder layer’s `linear_attn` from a local Qwen3.5 snapshot. |
| `bench_gdn_forward_random.py` | Random `torch.nn.init` | Inlined **Qwen3.5-0.8B / 9B**-shaped configs; no weight files. |

**Reference test environment:** Docker image **`quay.io/ascend/vllm-ascend:v0.18.0rc1`**

## Example commands

From this directory:

```bash
# Real weights (set --model to your local Qwen3.5 snapshot with config + safetensors)
python3 bench_gdn_forward.py \
  --model /scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/<revision>/ \
  --device 0 \
  --shapes 1x4096 4x4096 8x2048 \
  --warmup 2 --repeats 7
```

```bash
# Random weights, 0.8B-class shapes (default)
python3 bench_gdn_forward_random.py --variant 0.8B --device 0 --warmup 2 --repeats 7

# Random weights, 9B-class shapes
python3 bench_gdn_forward_random.py --variant 9B --device 0 --warmup 2 --repeats 7

# Random weights, longer sequences (example)
python3 bench_gdn_forward_random.py --variant 0.8B --shapes 1x8192 4x8192 1x16384 --warmup 2 --repeats 5
python3 bench_gdn_forward_random.py --variant 9B   --shapes 1x8192 4x8192 1x16384 --warmup 2 --repeats 5
```

Optional: `--verbose-logs` on either script for full vLLM INFO logs (default is quieter).

## Reference output (illustrative)

Captured in **`quay.io/ascend/vllm-ascend:v0.18.0rc1`** on a **910B2** NPU, **`bfloat16`**, layer **0**. Default-shape tables use **`--warmup 2 --repeats 7`**, shapes **`1x4096 4x4096 8x2048`**. Larger-shape rows use **`--warmup 2 --repeats 5`**. Your median ms and derived TFLOP/s / GiB/s will differ by chip, driver, and load.

**`bench_gdn_forward.py`** (real Qwen3.5-0.8B weights):

```
shape batch×seq=1×4096 (T=4096)   | median ~7.89 ms  | ~15.3 TFLOP/s (GEMM+proxy) | ~16.4 GiB/s (model bytes)
shape batch×seq=4×4096 (T=16384) | median ~14.76 ms | ~32.7 TFLOP/s (GEMM+proxy) | ~31.1 GiB/s (model bytes)
shape batch×seq=8×2048 (T=16384) | median ~14.49 ms | ~33.3 TFLOP/s (GEMM+proxy) | ~31.6 GiB/s (model bytes)
```

**`bench_gdn_forward_random.py --variant 0.8B`** (same default shapes; random weights):

```
shape batch×seq=1×4096 (T=4096)   | median ~7.78 ms  | ~15.5 TFLOP/s (GEMM+proxy) | ~16.6 GiB/s
shape batch×seq=4×4096 (T=16384) | median ~14.59 ms | ~33.1 TFLOP/s (GEMM+proxy) | ~31.4 GiB/s
shape batch×seq=8×2048 (T=16384) | median ~14.33 ms | ~33.7 TFLOP/s (GEMM+proxy) | ~32.0 GiB/s
```

**`bench_gdn_forward_random.py --variant 0.8B`** (larger shapes; random weights; `--repeats 5`):

```
shape batch×seq=1×8192  (T=8192)  | median ~10.53 ms | ~22.9 TFLOP/s (GEMM+proxy) | ~22.7 GiB/s
shape batch×seq=4×8192  (T=32768) | median ~66.07 ms | ~14.6 TFLOP/s (GEMM+proxy) | ~13.6 GiB/s
shape batch×seq=1×16384 (T=16384) | median ~17.40 ms | ~27.8 TFLOP/s (GEMM+proxy) | ~26.3 GiB/s
```

**`bench_gdn_forward_random.py --variant 9B`** (default shapes; random weights):

```
shape batch×seq=1×4096 (T=4096)   | median ~11.87 ms | ~52.3 TFLOP/s (GEMM+proxy) | ~29.0 GiB/s
shape batch×seq=4×4096 (T=16384) | median ~30.82 ms | ~80.6 TFLOP/s (GEMM+proxy) | ~32.5 GiB/s
shape batch×seq=8×2048 (T=16384) | median ~30.09 ms | ~82.5 TFLOP/s (GEMM+proxy) | ~33.3 GiB/s
```

**`bench_gdn_forward_random.py --variant 9B`** (larger shapes; random weights; `--repeats 5`):

```
shape batch×seq=1×8192  (T=8192)  | median ~18.79 ms | ~66.1 TFLOP/s (GEMM+proxy) | ~30.0 GiB/s
shape batch×seq=4×8192  (T=32768) | median ~96.37 ms | ~51.5 TFLOP/s (GEMM+proxy) | ~19.5 GiB/s
shape batch×seq=1×16384 (T=16384) | median ~34.70 ms | ~71.6 TFLOP/s (GEMM+proxy) | ~28.9 GiB/s
```

Median latency between **`bench_gdn_forward.py`** and **`bench_gdn_forward_random.py --variant 0.8B`** on default shapes typically agrees within **~1–2%**; TFLOP/s and GiB/s use the scripts’ analytical estimates, not hardware counters.
