#!/usr/bin/env python3
"""
Compare **Triton** vs **PTO** ``chunk_gated_delta_rule`` on lightweight metrics
(vLLM-native; no ``lm_eval`` VLLM wrapper).

**Wikitext-2 (validation)** — token-level perplexity on a bounded prefix.

**MMLU** (optional) — 5-shot MC accuracy via ``prompt_logprobs`` on choice spans.

**No ``subprocess.run`` in this file:** vLLM already spawns workers. Use one Python
process per backend (``record``), then ``compare`` on the two JSON files — or
``./run_compare_lm_eval.sh`` / ``./sanity_pto_triton_eval.sh``.

Dependencies: ``datasets`` (``pip install datasets``).

Examples::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 compare_pto_triton_lm_eval.py record --backend triton --output ./tmp/tri.json \
        --wiki-max-pages 1 --wiki-window 128 --max-model-len 768 --skip-mmlu
    python3 compare_pto_triton_lm_eval.py record --backend pto --output ./tmp/pto.json \
        --wiki-max-pages 1 --wiki-window 128 --max-model-len 768 --skip-mmlu
    python3 compare_pto_triton_lm_eval.py compare ./tmp/tri.json ./tmp/pto.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_PATCH = Path(__file__).resolve().parent

from compare_prefill_next_token import _apply_record_environ, _verify_chunk_backend


def _wikitext_detokenizer(page: str) -> str:
    string = page.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    string = (
        string.replace(" : ", ": ")
        .replace(" ; ", "; ")
        .replace(" . ", ". ")
        .replace(" ! ", "! ")
        .replace(" ? ", "? ")
        .replace(" , ", ", ")
    )
    string = string.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    string = string.replace(" \n", "\n").replace("\n ", "\n")
    string = string.replace(" N ", " 1 ").replace(" 's", "'s")
    return string


def _sum_prompt_nll(
    prompt_logprobs: Sequence[dict[int, Any] | None],
    token_ids: list[int],
) -> tuple[float, int]:
    total = 0.0
    n = 0
    for i, tid in enumerate(token_ids):
        if i == 0:
            continue
        lpdict = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        if not lpdict:
            continue
        ent = lpdict.get(int(tid))
        if ent is None:
            continue
        lp = float(getattr(ent, "logprob", ent))
        total -= lp
        n += 1
    return total, n


@dataclass
class MMLUExample:
    subject: str
    question: str
    choices: list[str]
    answer_idx: int


MMLU_SUBJECTS_ALL: tuple[str, ...] = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def _load_mmlu_examples(
    subjects: list[str] | None,
    max_samples: int | None,
    num_fewshot: int,
    seed: int,
) -> tuple[list[MMLUExample], dict[str, list[dict[str, Any]]]]:
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    fewshot_by_subject: dict[str, list[dict[str, Any]]] = {}
    test_rows: list[MMLUExample] = []

    subj_list = subjects if subjects else list(MMLU_SUBJECTS_ALL)

    for subj in subj_list:
        dev = load_dataset("cais/mmlu", subj, split="dev")
        idx = rng.choice(len(dev), size=min(num_fewshot, len(dev)), replace=False)
        fewshot_by_subject[subj] = [dev[int(i)] for i in idx]
        test = load_dataset("cais/mmlu", subj, split="test")
        for row in test:
            test_rows.append(
                MMLUExample(
                    subject=subj,
                    question=row["question"],
                    choices=list(row["choices"]),
                    answer_idx=int(ord(row["answer"]) - ord("A")),
                )
            )

    rng.shuffle(test_rows)
    if max_samples is not None:
        test_rows = test_rows[: max_samples]
    return test_rows, fewshot_by_subject


def _mmlu_prompt(ex: MMLUExample, fewshot_rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for fs in fewshot_rows:
        q = fs["question"].strip()
        ch = fs["choices"]
        ans = fs["answer"]
        parts.append(
            f"Q: {q}\n(A) {ch[0]} (B) {ch[1]} (C) {ch[2]} (D) {ch[3]}\nA: {ans}\n\n"
        )
    q = ex.question.strip()
    parts.append(
        f"Q: {q}\n(A) {ex.choices[0]} (B) {ex.choices[1]} (C) {ex.choices[2]} (D) {ex.choices[3]}\nA:"
    )
    return "".join(parts)


def _choice_token_suffixes(tokenizer: Any, letters: tuple[str, ...] = ("A", "B", "C", "D")) -> list[list[int]]:
    out: list[list[int]] = []
    for L in letters:
        ids = tokenizer.encode(L, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"empty tokenization for choice letter {L!r}")
        out.append(ids)
    return out


@dataclass
class EvalConfig:
    model: str
    max_model_len: int
    max_logprobs: int
    wiki_max_pages: int
    wiki_window: int
    skip_mmlu: bool
    mmlu_max_samples: int | None
    mmlu_subjects: list[str]
    num_fewshot: int
    seed: int
    max_num_seqs: int


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    torch.npu.set_device("npu:0")
    _verify_chunk_backend()

    llm = LLM(
        model=cfg.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_logprobs=cfg.max_logprobs,
        max_num_batched_tokens=cfg.max_model_len + 2048,
        max_num_seqs=cfg.max_num_seqs,
    )
    tok = llm.get_tokenizer()
    choice_suffixes = _choice_token_suffixes(tok)

    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="validation")
    all_ids: list[int] = []
    for i in range(min(cfg.wiki_max_pages, len(ds))):
        text = _wikitext_detokenizer(ds[i]["page"])
        if not text.strip():
            continue
        all_ids.extend(tok.encode(text, add_special_tokens=False))
        if len(all_ids) >= cfg.wiki_window * max(8, cfg.wiki_max_pages):
            break

    win = min(cfg.wiki_window, cfg.max_model_len - 2)
    if win < 8:
        raise ValueError(f"wiki_window too large for max_model_len-2 ({cfg.max_model_len})")

    sp_wiki = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        prompt_logprobs=-1,
        seed=cfg.seed,
    )
    nll_sum = 0.0
    n_tok = 0
    for start in range(0, len(all_ids) - 1, win):
        chunk = all_ids[start : start + win]
        if len(chunk) < 2:
            continue
        out = llm.generate([TokensPrompt(prompt_token_ids=chunk)], sp_wiki)[0]
        pl = out.prompt_logprobs
        if pl is None:
            raise RuntimeError("wikitext: expected prompt_logprobs")
        s, n = _sum_prompt_nll(pl, chunk)
        nll_sum += s
        n_tok += n

    wiki_ppl = math.exp(nll_sum / max(n_tok, 1))

    if cfg.skip_mmlu:
        mmlu_acc = float("nan")
        examples: list[MMLUExample] = []
    else:
        examples, fewshot_map = _load_mmlu_examples(
            cfg.mmlu_subjects if cfg.mmlu_subjects else None,
            cfg.mmlu_max_samples,
            num_fewshot=cfg.num_fewshot,
            seed=cfg.seed,
        )
        correct = 0
        sp_mmlu = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            prompt_logprobs=-1,
            seed=cfg.seed + 1,
        )

        for ex in examples:
            prefix = _mmlu_prompt(ex, fewshot_map[ex.subject])
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            scores = []
            for ci in range(4):
                suff = choice_suffixes[ci]
                full = prefix_ids + suff
                out = llm.generate([TokensPrompt(prompt_token_ids=full)], sp_mmlu)[0]
                pl = out.prompt_logprobs
                if pl is None:
                    raise RuntimeError("mmlu: expected prompt_logprobs")
                start_i = len(prefix_ids)
                ll_sum = 0.0
                for j in range(start_i, len(full)):
                    tid = full[j]
                    lpdict = pl[j] if j < len(pl) else None
                    if not lpdict or tid not in lpdict:
                        ll_sum += -80.0
                    else:
                        ll_sum += float(getattr(lpdict[tid], "logprob", lpdict[tid]))
                scores.append(ll_sum)
            pred = int(np.argmax(scores))
            if pred == ex.answer_idx:
                correct += 1

        mmlu_acc = correct / max(len(examples), 1)

    backend = os.environ.get("_CMP_BACKEND", "")
    return {
        "backend": backend,
        "wikitext_token_ppl": wiki_ppl,
        "wikitext_n_tokens_scored": n_tok,
        "wikitext_nll_sum": nll_sum,
        "mmlu_acc": mmlu_acc,
        "mmlu_n": len(examples) if not cfg.skip_mmlu else 0,
        "mmlu_skipped": bool(cfg.skip_mmlu),
    }


def cmd_record(args: argparse.Namespace) -> int:
    _apply_record_environ(backend=args.backend, device=args.device)
    if args.skip_mmlu:
        os.environ["EVAL_SKIP_MMLU"] = "1"
    else:
        os.environ.pop("EVAL_SKIP_MMLU", None)

    subj = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()] if args.mmlu_subjects else []

    cfg = EvalConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_logprobs=args.max_logprobs,
        wiki_max_pages=args.wiki_max_pages,
        wiki_window=args.wiki_window,
        skip_mmlu=args.skip_mmlu,
        mmlu_max_samples=args.mmlu_max_samples,
        mmlu_subjects=subj,
        num_fewshot=args.num_fewshot,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
    )
    summary = run_eval(cfg)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(outp), **summary}, indent=2), flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    tri = json.loads(Path(args.triton_json).read_text(encoding="utf-8"))
    pto = json.loads(Path(args.pto_json).read_text(encoding="utf-8"))

    wt, wp = float(tri["wikitext_token_ppl"]), float(pto["wikitext_token_ppl"])
    print("\n=== Wikitext token PPL ===", flush=True)
    print(f"Triton: {wt:.6g}   PTO: {wp:.6g}   n_tokens={tri['wikitext_n_tokens_scored']}", flush=True)
    rel = abs(wp - wt) / max(wt, 1e-8)
    print(f"Relative diff: {rel:.6g} (rtol={args.wiki_ppl_rtol})", flush=True)
    ok = rel <= args.wiki_ppl_rtol
    if not ok:
        print("FAIL: Wikitext PPL rtol.", flush=True)

    if tri.get("mmlu_skipped") or pto.get("mmlu_skipped"):
        print("MMLU skipped in one or both runs; skipping acc check.", flush=True)
        return 0 if ok else 1

    mt, mp = float(tri["mmlu_acc"]), float(pto["mmlu_acc"])
    print("\n=== MMLU acc ===", flush=True)
    print(f"Triton: {mt:.6g}   PTO: {mp:.6g}   n={tri['mmlu_n']}", flush=True)
    adiff = abs(mp - mt)
    print(f"Abs diff: {adiff:.6g} (atol={args.mmlu_acc_atol})", flush=True)
    if adiff > args.mmlu_acc_atol:
        print("FAIL: MMLU atol.", flush=True)
        ok = False
    else:
        print("OK: MMLU within atol.", flush=True)

    if ok:
        print("\nOK: all checks passed.", flush=True)
    return 0 if ok else 1


def _add_record_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default="/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    )
    p.add_argument("--device", default="0", help="ASCEND_RT_VISIBLE_DEVICES (single NPU index)")
    p.add_argument("--backend", choices=("triton", "pto"), required=True)
    p.add_argument("--output", type=Path, required=True, help="Output metrics JSON path")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-logprobs", type=int, default=300_000)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--wiki-max-pages", type=int, default=20)
    p.add_argument("--wiki-window", type=int, default=2048)
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--mmlu-max-samples", type=int, default=None)
    p.add_argument("--mmlu-subjects", default="", help="Comma-separated subject names; empty = all (slow)")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)


def main() -> int:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("record", help="Run one backend; write metrics JSON (fresh Python process per backend).")
    _add_record_args(p_rec)
    p_rec.set_defaults(func=cmd_record)

    p_cmp = sub.add_parser("compare", help="Compare two JSON files from record (no vLLM import).")
    p_cmp.add_argument("triton_json", type=str)
    p_cmp.add_argument("pto_json", type=str)
    p_cmp.add_argument("--wiki-ppl-rtol", type=float, default=0.05)
    p_cmp.add_argument("--mmlu-acc-atol", type=float, default=0.03)
    p_cmp.set_defaults(func=cmd_compare)

    args = root.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        raise SystemExit("Missing dependency (try: pip install datasets).\n" + str(e)) from e
