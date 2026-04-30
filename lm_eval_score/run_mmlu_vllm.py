#!/usr/bin/env python3
"""Run lm-eval tasks via the offline vLLM engine on Ascend NPU.

Default tasks: **six MMLU subjects** (fast subset), **GPQA Diamond** (zeroshot MC),
**wikitext** (perplexity). Use ``--full-mmlu`` for the full ``mmlu`` task group.
Override tasks with ``--tasks``.

Model presets match paths in ``qwen35_prefill/profile_qwen35_prefill.py`` plus the
local MoE W8A8 checkpoints.

Default is **single NPU**, ``tensor_parallel_size=1`` (set ``ASCEND_RT_VISIBLE_DEVICES``).

Timing (written with ``--output-json``): wall-clock total; ``vllm_llm_init_seconds``
(time inside ``vLLM.LLM.__init__``, mainly weights/engine); ``lm_eval_model_setup_seconds``
(full lm-eval ``VLLM`` wrapper init including tokenizer); ``eval_execution_seconds``
(harness running tasks after the model object is ready). During eval, vLLM tqdm lines
report ``est. speed input: … toks/s`` — the JSON records **last** and **peak** of those
running averages (see ``timing`` keys ``*_toks_per_s_*``).

Examples::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 run_mmlu_vllm.py --preset qwen36_27b_w8a8 \\
        --output-json outputs/qwen27_suite.json

    python3 run_mmlu_vllm.py --preset qwen35_0_8b --full-mmlu --output-json /tmp/full.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lm_eval
from lm_eval.utils import handle_non_serializable, make_table

DEFAULT_MODEL = "/scratch/model_weights/Qwen3.6-35B-A3B-w8a8"

# Representative MMLU slice (STEM / humanities / social / professional) — much faster than group ``mmlu``.
DEFAULT_MMLU_SUBSET = (
    "mmlu_astronomy,"
    "mmlu_high_school_mathematics,"
    "mmlu_college_biology,"
    "mmlu_high_school_world_history,"
    "mmlu_professional_law,"
    "mmlu_philosophy"
)
DEFAULT_TASKS_SUBSET = f"{DEFAULT_MMLU_SUBSET},gpqa_diamond_zeroshot,wikitext"
FULL_TASKS_WITH_MMLU_GROUP = "mmlu,gpqa_diamond_zeroshot,wikitext"
_AUTO_QUANT = "__AUTO_QUANT__"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass(frozen=True)
class ModelPreset:
    key: str
    path: str
    quantization: str | None
    expert_parallel: bool


MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen35_0_8b": ModelPreset(
        "qwen35_0_8b",
        "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
        None,
        False,
    ),
    "qwen35_9b": ModelPreset(
        "qwen35_9b",
        "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        None,
        False,
    ),
    "qwen36_27b_w8a8": ModelPreset(
        "qwen36_27b_w8a8",
        "/scratch/model_weights/Qwen3.6-27B-w8a8",
        "ascend",
        False,
    ),
    "qwen36_35b_a3b_w8a8": ModelPreset(
        "qwen36_35b_a3b_w8a8",
        "/scratch/model_weights/Qwen3.6-35B-A3B-w8a8",
        "ascend",
        True,
    ),
}


def _patch_autoconfig_for_local_qwen35_checkpoints(model_path: str):
    """lm-eval's ``VLLM`` wrapper calls ``AutoConfig.from_pretrained``.

    Local Qwen3.6 checkpoints may use ``model_type`` values Transformers has not
    registered yet (``qwen3_5``, ``qwen3_5_moe``). Map ``text_config`` onto a
    known ``PretrainedConfig`` class so tokenizer helpers work; vLLM still loads
    the real checkpoint.

    Returns a zero-arg restore callback (no-op if patch was not applied).
    """

    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    resolved = Path(model_path).expanduser().resolve()
    cfg_file = resolved / "config.json"
    if not cfg_file.is_file():
        return lambda: None

    try:
        meta = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return lambda: None

    mt = meta.get("model_type")
    if mt == "qwen3_5_moe":

        def build_cfg():
            tc = dict(meta.get("text_config") or {})
            tc["model_type"] = "qwen3_moe"
            return Qwen3MoeConfig(**tc)

    elif mt == "qwen3_5":

        def build_cfg():
            tc = dict(meta.get("text_config") or {})
            tc["model_type"] = "qwen3"
            return Qwen3Config(**tc)

    else:
        return lambda: None

    orig_fn = AutoConfig.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            p = Path(pretrained_model_name_or_path).expanduser().resolve()
        except TypeError:
            return orig_fn(cls, pretrained_model_name_or_path, *args, **kwargs)
        if p != resolved:
            return orig_fn(cls, pretrained_model_name_or_path, *args, **kwargs)
        return build_cfg()

    AutoConfig.from_pretrained = _patched_from_pretrained

    def _restore() -> None:
        AutoConfig.from_pretrained = classmethod(orig_fn)

    return _restore


def _resolve_quantization(args: argparse.Namespace, model_path: str) -> str | None:
    if getattr(args, "no_quantization", False):
        return None
    if args.quantization != _AUTO_QUANT:
        return args.quantization or None
    if args.preset:
        return MODEL_PRESETS[args.preset].quantization
    low = model_path.lower()
    if "w8a8" in low or "w4a8" in low or "quant_model" in low:
        return "ascend"
    return None


def _resolve_expert_parallel(args: argparse.Namespace) -> bool:
    if args.no_expert_parallel:
        return False
    if args.preset:
        return MODEL_PRESETS[args.preset].expert_parallel
    # Without a preset, avoid enabling EP on dense checkpoints (vLLM rejects EP when num_experts==0).
    mp = (args.model or "").lower()
    if "a3b" in mp or "moe" in mp:
        return True
    return False


def build_model_args(ns: argparse.Namespace) -> dict[str, object]:
    args: dict[str, object] = {
        "pretrained": ns.model,
        "trust_remote_code": True,
        "max_model_len": ns.max_model_len,
        "tensor_parallel_size": ns.tensor_parallel_size,
        "dtype": ns.dtype,
        "enforce_eager": ns.enforce_eager,
        "gpu_memory_utilization": ns.gpu_memory_utilization,
    }
    if ns.quantization:
        args["quantization"] = ns.quantization
    if ns.enable_expert_parallel:
        args["enable_expert_parallel"] = True
    return args


def _install_eval_timing_hooks() -> tuple[dict[str, float], Callable[[], None]]:
    """Measure vLLM engine init vs full lm-eval VLLM wrapper init."""

    import vllm
    from lm_eval.models.vllm_causallms import VLLM as LMEvalVLLM

    llm_orig = vllm.LLM.__init__
    wrap_orig = LMEvalVLLM.__init__

    timings: dict[str, float] = {
        "vllm_llm_init_seconds": 0.0,
        "lm_eval_model_setup_seconds": 0.0,
    }

    def llm_wrapped(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return llm_orig(self, *a, **kw)
        finally:
            timings["vllm_llm_init_seconds"] = time.perf_counter() - t0

    def wrap_wrapped(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return wrap_orig(self, *a, **kw)
        finally:
            timings["lm_eval_model_setup_seconds"] = time.perf_counter() - t0

    vllm.LLM.__init__ = llm_wrapped  # type: ignore[method-assign]
    LMEvalVLLM.__init__ = wrap_wrapped  # type: ignore[method-assign]

    def restore() -> None:
        vllm.LLM.__init__ = llm_orig  # type: ignore[method-assign]
        LMEvalVLLM.__init__ = wrap_orig  # type: ignore[method-assign]

    return timings, restore


_POSTFIX_TP_RE = re.compile(r"est\. speed input: ([\d.]+) toks/s, output: ([\d.]+) toks/s")


def _patch_vllm_processed_prompts_throughput(
    store: dict[str, float | None],
) -> Callable[[], None]:
    """Capture vLLM ``Processed prompts`` tqdm postfix (input/output toks/s).

    Patches ``vllm.entrypoints.llm.tqdm`` (the symbol ``LLM._run_engine`` closes over).
    """

    import vllm.entrypoints.llm as llm_entry

    Orig = llm_entry.tqdm

    class _TqdmThroughput(Orig):
        def update(self, n=1):  # type: ignore[override]
            r = super().update(n)
            self._snap_throughput()
            return r

        def refresh(self, *args, **kwargs):  # type: ignore[override]
            r = super().refresh(*args, **kwargs)
            self._snap_throughput()
            return r

        def close(self, *args, **kwargs):  # type: ignore[override]
            self._snap_throughput()
            return super().close(*args, **kwargs)

        def _snap_throughput(self) -> None:
            pf = getattr(self, "postfix", None)
            if not isinstance(pf, str):
                return
            m = _POSTFIX_TP_RE.search(pf)
            if not m:
                return
            inp, outp = float(m.group(1)), float(m.group(2))
            store["last_input_toks_per_s"] = inp
            store["last_output_toks_per_s"] = outp
            pi = store.get("peak_input_toks_per_s")
            po = store.get("peak_output_toks_per_s")
            store["peak_input_toks_per_s"] = inp if pi is None else max(pi, inp)
            store["peak_output_toks_per_s"] = outp if po is None else max(po, outp)

    llm_entry.tqdm = _TqdmThroughput  # type: ignore[misc]

    def restore() -> None:
        llm_entry.tqdm = Orig

    return restore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default=None,
        help="Known model checkpoint (sets path, quantization, expert-parallel defaults)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override checkpoint path (optional if --preset is set)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated lm-eval tasks (default: 6 MMLU subjects + gpqa_diamond_zeroshot + wikitext; "
            "use --full-mmlu for entire mmlu group)"
        ),
    )
    parser.add_argument(
        "--full-mmlu",
        action="store_true",
        help="Replace default MMLU subset with the full ``mmlu`` task group (very slow)",
    )
    parser.add_argument(
        "--skip-gpqa-diamond",
        action="store_true",
        help="Drop gpqa_diamond_zeroshot (Hub dataset Idavidrein/gpqa is gated — needs HF_TOKEN)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Single-device default: 1 (no tensor parallelism)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=(
            int(os.environ["LM_EVAL_MAX_MODEL_LEN"])
            if "LM_EVAL_MAX_MODEL_LEN" in os.environ
            else 8192
        ),
        help="Context length for vLLM (lower helps Wikitext rolling + prompt_logprobs peak memory)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=(
            float(os.environ["LM_EVAL_GPU_MEMORY_UTILIZATION"])
            if "LM_EVAL_GPU_MEMORY_UTILIZATION" in os.environ
            else 0.85
        ),
        help="vLLM KV/cache reservation fraction (lower if Wikitext rolling loglikelihood OOMs)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.environ["LM_EVAL_MAX_BATCH_SIZE"])
        if "LM_EVAL_MAX_BATCH_SIZE" in os.environ
        else None,
        help="Forwarded to lm-eval simple_evaluate as max_batch_size (caps auto batching)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=_AUTO_QUANT,
        metavar="NAME",
        help="vLLM quantization backend; default follows --preset / path heuristic",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Do not pass quantization= to vLLM (dense BF16/FP16 checkpoints)",
    )
    parser.add_argument(
        "--no-expert-parallel",
        action="store_true",
        help="Disable MoE expert parallelism",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eager mode avoids graph capture (default: True)",
    )
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Cap examples per task (int count or fraction in (0,1))",
    )
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=100000,
        help="Bootstrap iterations for stderr; 0 skips stderr",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write results + timing + lm-eval tables sidecar",
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="vLLM seed and lm-eval RNG seeds",
    )
    args = parser.parse_args()

    if args.tasks is not None:
        task_string = args.tasks
    elif args.full_mmlu:
        task_string = FULL_TASKS_WITH_MMLU_GROUP
    else:
        task_string = DEFAULT_TASKS_SUBSET

    if args.preset:
        model_path = MODEL_PRESETS[args.preset].path
    elif args.model:
        model_path = args.model
    else:
        model_path = DEFAULT_MODEL

    task_list = [t.strip() for t in task_string.split(",") if t.strip()]
    if args.skip_gpqa_diamond:
        task_list = [t for t in task_list if t != "gpqa_diamond_zeroshot"]
    if not task_list:
        parser.error("No tasks left after filters (--tasks / --skip-gpqa-diamond)")

    quant = _resolve_quantization(args, model_path)
    enable_ep = _resolve_expert_parallel(args)

    ns = argparse.Namespace(
        model=model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=quant,
        enable_expert_parallel=enable_ep,
    )
    model_args = build_model_args(ns)
    model_args["seed"] = args.seed

    print(
        "model_args:",
        model_args,
        "preset=",
        args.preset,
        "tasks=",
        task_list,
        flush=True,
    )

    _restore_autoconfig = _patch_autoconfig_for_local_qwen35_checkpoints(model_path)
    _timings, _restore_hooks = _install_eval_timing_hooks()
    tp_store: dict[str, float | None] = {}
    _restore_tp = _patch_vllm_processed_prompts_throughput(tp_store)
    wall_t0 = time.perf_counter()
    results = None
    eval_exc: BaseException | None = None
    try:
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=task_list,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
            bootstrap_iters=args.bootstrap_iters,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
    except BaseException as exc:
        eval_exc = exc
    finally:
        wall_total = time.perf_counter() - wall_t0
        _restore_hooks()
        _restore_tp()
        _restore_autoconfig()

    if eval_exc is not None:
        raise eval_exc

    setup_s = _timings["lm_eval_model_setup_seconds"]
    llm_s = _timings["vllm_llm_init_seconds"]

    def _round2_opt(val: float | None) -> float | None:
        return None if val is None else round(float(val), 2)

    timing = {
        "wall_clock_total_seconds": round(wall_total, 4),
        "vllm_llm_init_seconds": round(llm_s, 4),
        "lm_eval_model_setup_seconds": round(setup_s, 4),
        "tokenizer_config_overhead_seconds": round(max(setup_s - llm_s, 0.0), 4),
        "eval_execution_seconds": round(max(wall_total - setup_s, 0.0), 4),
        "eval_est_speed_input_toks_per_s_last": _round2_opt(tp_store.get("last_input_toks_per_s")),
        "eval_est_speed_output_toks_per_s_last": _round2_opt(tp_store.get("last_output_toks_per_s")),
        "eval_est_speed_input_toks_per_s_peak": _round2_opt(tp_store.get("peak_input_toks_per_s")),
        "eval_est_speed_output_toks_per_s_peak": _round2_opt(tp_store.get("peak_output_toks_per_s")),
        "eval_throughput_note": (
            "From vLLM tqdm postfix during Processed prompts (running average "
            "tot_tokens/elapsed per llm.py; peaks accumulate across batches)."
        ),
    }
    print("timing:", timing, flush=True)

    print(make_table(results))
    if results.get("groups"):
        print(make_table(results, "groups"))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        mmlu_mode = (
            "full_mmlu_group"
            if args.full_mmlu and args.tasks is None
            else ("custom_tasks" if args.tasks is not None else "mmlu_subset_6")
        )
        payload = {
            "preset": args.preset,
            "model_path": model_path,
            "model_args": model_args,
            "tasks": task_list,
            "task_string": task_string,
            "gpqa_diamond_included": "gpqa_diamond_zeroshot" in task_list,
            "limit": args.limit,
            "seed": args.seed,
            "timing": timing,
            "results": results["results"],
            "groups": results.get("groups"),
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=handle_non_serializable)
        tables_path = args.output_json.with_name(args.output_json.stem + "_tables.txt")
        with tables_path.open("w", encoding="utf-8") as f:
            f.write(f"timing_seconds: {json.dumps(timing)}\n")
            f.write(
                "vLLM_tqdm_est_speed_toks_per_s: "
                f"last_input={timing['eval_est_speed_input_toks_per_s_last']}, "
                f"last_output={timing['eval_est_speed_output_toks_per_s_last']}, "
                f"peak_input={timing['eval_est_speed_input_toks_per_s_peak']}, "
                f"peak_output={timing['eval_est_speed_output_toks_per_s_peak']}\n\n"
            )
            f.write(make_table(results))
            f.write("\n")
            if results.get("groups"):
                f.write(make_table(results, "groups"))
                f.write("\n")


if __name__ == "__main__":
    main()
