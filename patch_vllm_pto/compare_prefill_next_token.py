#!/usr/bin/env python3
"""Compare first decoded token (greedy) with vs without PTO patch — end-to-end smoke test."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PATCH = Path(__file__).resolve().parent


def _child_payload() -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = os.environ["_CMP_MODEL"]
    # With ASCEND_RT_VISIBLE_DEVICES set to a single physical id, the process sees it as npu:0.
    torch.npu.set_device("npu:0")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    seed = "The quick brown fox jumps over the lazy dog. "
    slen = int(os.environ["_CMP_SEQ"])
    ids: list[int] = []
    while len(ids) < slen:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    ids = ids[:slen]
    prompt = tok.decode(ids)
    prompts = [prompt]

    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(slen + 32, 4096),
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=slen + 128,
        max_num_seqs=8,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)
    out = llm.generate(prompts, sp)[0]
    tid = int(out.outputs[0].token_ids[0])
    print(json.dumps({"first_token_id": tid}))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default="4", help="ASCEND_RT_VISIBLE_DEVICES (single NPU index)")
    args = p.parse_args()

    env_base = os.environ.copy()
    env_base["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env_base["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    env_base["_CMP_MODEL"] = args.model
    env_base["_CMP_SEQ"] = str(args.seq_len)

    script = str(Path(__file__).resolve())

    def _run(with_pto: bool) -> int:
        env = env_base.copy()
        if with_pto:
            env["VLLM_PTO_PATCH_DIR"] = str(_PATCH)
        else:
            env.pop("VLLM_PTO_PATCH_DIR", None)
        r = subprocess.run(
            [sys.executable, script, "--internal-child"],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if r.returncode != 0:
            print(r.stderr[-8000:])
            raise RuntimeError(f"subprocess failed rc={r.returncode}")
        lines = [x for x in r.stdout.splitlines() if x.strip().startswith("{")]
        if not lines:
            print(r.stdout)
            raise RuntimeError("no JSON line in child stdout")
        return json.loads(lines[-1])["first_token_id"]

    t0 = _run(False)
    t1 = _run(True)
    print(f"Triton baseline first_token_id={t0}  PTO-patched first_token_id={t1}")
    if t0 != t1:
        print("MISMATCH: greedy next token differs.")
        return 1
    print("OK: greedy next token matches.")
    return 0


if __name__ == "__main__":
    if "--internal-child" in sys.argv:
        _child_payload()
    else:
        raise SystemExit(main())
