#!/usr/bin/env python3
"""Sweep Qwen3.5 prefill profiling (eager only): 9B then 0.8B, fixed (batch, seq) grid.

Runs ``profile_qwen35_prefill.py`` per case, logs end-to-end time, zips each trace tree,
and writes ``qwen35_profile_summary.md`` from ``op_statistic.csv`` results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Same snapshot paths as profile_qwen35_prefill.py (lines 12–14).
MODEL_0_8B = (
    "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17/"
)
MODEL_9B = (
    "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a/"
)

# 9B first, then 0.8B; eager-only sweep uses these (batch, seq_len) pairs.
SWEEP_MODELS: list[tuple[str, str]] = [
    ("9B", MODEL_9B),
    ("0.8B", MODEL_0_8B),
]
# Avoid large batch × long seq (huge traces); similar token corners.
SWEEP_BATCH_SEQ_PAIRS: list[tuple[int, int]] = [
    (1, 4096),
    (1, 16384),
    (2, 8192),
    (16, 1024),
]

PROFILE_SCRIPT = Path(__file__).resolve().parent / "profile_qwen35_prefill.py"
SUMMARY_MD = Path(__file__).resolve().parent / "qwen35_profile_summary.md"
RESULTS_JSON = "sweep_results.json"
MARKER_OK = ".profile_case_ok"


@dataclass
class CaseResult:
    case_id: str
    model_label: str
    model_path: str
    eager: bool
    batch_size: int
    seq_len: int
    profile_dir: str
    status: str
    duration_sec: float | None
    error_snippet: str | None
    zip_path: str | None


def _seq_label(seq_len: int) -> str:
    if seq_len % 1024 == 0:
        return f"{seq_len // 1024}K"
    return str(seq_len)


def _case_dir_name(model_label: str, eager: bool, batch: int, seq_len: int) -> str:
    mode = "eager" if eager else "graph"
    return f"qwen35_{model_label}_{mode}_b{batch}_sl{_seq_label(seq_len)}"


def _is_oom(text: str) -> bool:
    t = text.lower()
    patterns = (
        "out of memory",
        "outofmemory",
        "oom",
        "allocator",
        "insufficient memory",
        "cuda error",
        "npu error",
    )
    return any(p in t for p in patterns)


def _find_op_statistic(profile_dir: Path) -> Path | None:
    candidates = sorted(
        profile_dir.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_top_ops(op_csv: Path, n: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with op_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Sort by Ratio(%) descending if parsable
    def ratio_key(r: dict[str, str]) -> float:
        key = "Ratio(%)" if "Ratio(%)" in r else next(
            (k for k in r if k.strip().lower().startswith("ratio")), "Ratio(%)"
        )
        try:
            return float(str(r.get(key, "0")).replace(",", ""))
        except ValueError:
            return 0.0

    rows.sort(key=ratio_key, reverse=True)
    out: list[dict[str, Any]] = []
    for r in rows[:n]:
        op_type = r.get("OP Type") or ""
        if not op_type and r:
            vals = list(r.values())
            op_type = vals[1] if len(vals) > 1 else ""
        ratio_k = next(
            (k for k in r if "ratio" in k.lower()),
            "Ratio(%)",
        )
        try:
            ratio = float(str(r.get(ratio_k, "0")).replace(",", ""))
        except ValueError:
            ratio = 0.0
        out.append({"op": op_type.strip(), "ratio_pct": ratio})
    return out


def _zip_profile_tree(profile_dir: Path, zip_path: Path) -> None:
    profile_dir = profile_dir.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    arcname = profile_dir.name
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path in sorted(profile_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(arcname) / path.relative_to(profile_dir)))


def _load_results_map(output_root: Path) -> dict[str, CaseResult]:
    path = output_root / RESULTS_JSON
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, CaseResult] = {}
    for row in rows:
        try:
            cr = CaseResult(**row)
            out[cr.case_id] = cr
        except (TypeError, KeyError):
            continue
    return out


def _ordered_case_results(
    output_root: Path,
    cases: list[tuple[str, str, bool, int, int]],
    results_map: dict[str, CaseResult],
) -> list[CaseResult]:
    out: list[CaseResult] = []
    for ml, mp, eager, b, sl in cases:
        cid = _case_dir_name(ml, eager, b, sl)
        if cid in results_map:
            out.append(results_map[cid])
            continue
        out.append(
            CaseResult(
                case_id=cid,
                model_label=ml,
                model_path=mp,
                eager=eager,
                batch_size=b,
                seq_len=sl,
                profile_dir=str(output_root / cid),
                status="not_run",
                duration_sec=None,
                error_snippet=None,
                zip_path=None,
            )
        )
    return out


def _merge_top_ops_from_disk(
    output_root: Path, top_ops_by_case: dict[str, list[dict[str, Any]]]
) -> None:
    for sub in sorted(output_root.glob("qwen35_*")):
        if not sub.is_dir():
            continue
        oc = _find_op_statistic(sub)
        if oc:
            top_ops_by_case[sub.name] = _read_top_ops(oc)


def _apply_skip_ok_case(
    output_root: Path,
    zips_dir: Path,
    ml: str,
    mp: str,
    eager: bool,
    b: int,
    sl: int,
    results_map: dict[str, CaseResult],
    top_ops_by_case: dict[str, list[dict[str, Any]]],
) -> bool:
    """If marker + zip exist, refresh results_map and top ops. Returns True if skipped."""
    case_id = _case_dir_name(ml, eager, b, sl)
    profile_dir = output_root / case_id
    zip_path = zips_dir / f"{case_id}_ascend_profile.zip"
    marker = profile_dir / MARKER_OK
    if not (marker.is_file() and zip_path.is_file()):
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    op_csv = _find_op_statistic(profile_dir)
    if op_csv:
        top_ops_by_case[case_id] = _read_top_ops(op_csv)
    prev = results_map.get(case_id)
    dur = data.get("duration_sec")
    if dur is None and prev and prev.duration_sec is not None:
        dur = prev.duration_sec
    results_map[case_id] = CaseResult(
        case_id=case_id,
        model_label=ml,
        model_path=mp,
        eager=eager,
        batch_size=b,
        seq_len=sl,
        profile_dir=str(profile_dir),
        status="ok",
        duration_sec=dur,
        error_snippet=None,
        zip_path=str(zip_path),
    )
    return True


def _top1_op(tops: list[dict[str, Any]] | None) -> tuple[str, float] | None:
    if not tops:
        return None
    return tops[0]["op"], float(tops[0]["ratio_pct"])


def _append_data_driven_operator_analysis(
    lines: list[str],
    results: list[CaseResult],
    top_ops_by_case: dict[str, list[dict[str, Any]]],
) -> None:
    """Append measured top-1 summary table (eager-only sweep)."""

    lines.append("### Sweep grid: top-1 operator per cell (eager, run order)")
    lines.append("")
    lines.append("| Model | Batch | Seq len | Case id | Top-1 op | Ratio % | Status |")
    lines.append("|---|:---:|---:|:---|:---|---:|:---|")
    for r in results:
        if r.status == "ok":
            t1 = _top1_op(top_ops_by_case.get(r.case_id))
            if t1:
                op, pct = t1
                lines.append(
                    f"| {r.model_label} | {r.batch_size} | {r.seq_len} | "
                    f"`{r.case_id}` | {op} | {pct:.3f} | ok |"
                )
            else:
                lines.append(
                    f"| {r.model_label} | {r.batch_size} | {r.seq_len} | "
                    f"`{r.case_id}` | — | — | ok (no op csv) |"
                )
        else:
            lines.append(
                f"| {r.model_label} | {r.batch_size} | {r.seq_len} | "
                f"`{r.case_id}` | — | — | {r.status} |"
            )
    lines.append("")


def _write_summary(
    output_root: Path,
    results: list[CaseResult],
    top_ops_by_case: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("# Qwen3.5 prefill profiling sweep")
    lines.append("")
    lines.append("This report is generated from `run_qwen35_profile_sweep.py` using Ascend ")
    lines.append("`op_statistic.csv` (top operators by total-time share).")
    lines.append("")
    n_ok = sum(1 for r in results if r.status == "ok")
    n_oom = sum(1 for r in results if r.status == "oom")
    n_err = sum(1 for r in results if r.status == "error")
    n_nr = sum(1 for r in results if r.status == "not_run")
    if n_nr or n_oom or n_err:
        lines.append(
            f"**Coverage:** {n_ok} ok, {n_oom} out-of-memory, {n_err} other errors, "
            f"{n_nr} not run. Run `python3 run_qwen35_profile_sweep.py` to execute the full "
            f"matrix ({len(results)} planned cases); use `--skip-ok` to resume after partial runs. "
            f"Regenerate this file with `--summary-only`."
        )
        lines.append("")

    lines.append("## End-to-end run times")
    lines.append("")
    lines.append("| Model | Mode | Batch | Seq len | Status | Time (s) | Notes |")
    lines.append("|---|:---:|---:|---:|:---:|---:|---|")
    for r in results:
        mode = "eager" if r.eager else "graph"
        note = ""
        if r.status == "oom":
            note = "Treated as out-of-memory (log heuristic)"
        elif r.status == "not_run":
            note = "Not executed yet (interrupted sweep or dry partial run)"
        elif r.status != "ok":
            note = (r.error_snippet or "")[:120]
        t = "" if r.duration_sec is None else f"{r.duration_sec:.2f}"
        lines.append(
            f"| {r.model_label} | {mode} | {r.batch_size} | {r.seq_len} | "
            f"{r.status} | {t} | {note} |"
        )
    lines.append("")

    lines.append("## Top-10 operators by case (from op_statistic.csv)")
    lines.append("")
    for r in results:
        if r.status != "ok":
            continue
        tops = top_ops_by_case.get(r.case_id)
        if not tops:
            continue
        mode = "eager" if r.eager else "graph"
        lines.append(f"### {r.case_id} ({r.model_label}, {mode}, batch={r.batch_size}, seq={r.seq_len})")
        lines.append("")
        lines.append("| Rank | Operator | Ratio (%) |")
        lines.append("|---:|:---|---:|")
        for i, row in enumerate(tops, start=1):
            lines.append(f"| {i} | {row['op']} | {row['ratio_pct']:.3f} |")
        lines.append("")

    lines.append("## How top operators shift with settings (data from this sweep)")
    lines.append("")
    _append_data_driven_operator_analysis(lines, results, top_ops_by_case)
    lines.append("### Most common operators among top-10 lists (by case count)")
    lines.append("")
    # Aggregate: for each op name, list (case_id, ratio) for top-10 membership
    op_presence: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for cid, tops in top_ops_by_case.items():
        for row in tops:
            op_presence[row["op"]].append((cid, row["ratio_pct"]))
    frequent = sorted(op_presence.items(), key=lambda x: (-len(x[1]), -max(t[1] for t in x[1])))[:15]
    lines.append("Ranked by how many successful cases include the operator in the top-10:")
    lines.append("")
    lines.append("")
    for op, lst in frequent:
        cases = ", ".join(f"{c} ({p:.1f}%)" for c, p in sorted(lst, key=lambda t: -t[1])[:5])
        if len(lst) > 5:
            cases += f", … (+{len(lst) - 5} cases)"
        lines.append(f"- **{op}**: {len(lst)} case(s) in top-10 — e.g. {cases}")
    lines.append("")

    lines.append("## Trace zip bundles")
    lines.append("")
    for r in results:
        if r.zip_path:
            lines.append(f"- `{r.zip_path}` — {r.case_id}")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "profile_sweep_runs",
        help="Root directory for per-case profile dirs and zips",
    )
    parser.add_argument(
        "--skip-ok",
        action="store_true",
        help="Skip cases that already have a success marker and zip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cases and exit",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild qwen35_profile_summary.md from sweep_results.json and on-disk traces "
        "(no vLLM runs)",
    )
    args = parser.parse_args()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    zips_dir = output_root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, str, bool, int, int]] = []
    for ml, mp in SWEEP_MODELS:
        for b, sl in SWEEP_BATCH_SEQ_PAIRS:
            cases.append((ml, mp, True, b, sl))

    if args.dry_run:
        for ml, mp, eager, b, sl in cases:
            cid = _case_dir_name(ml, eager, b, sl)
            print(cid, mp, "eager" if eager else "graph", b, sl)
        print(f"Total cases: {len(cases)}")
        return

    if args.summary_only:
        results_map = _load_results_map(output_root)
        top_ops_by_case: dict[str, list[dict[str, Any]]] = {}
        for ml, mp, eager, b, sl in cases:
            _apply_skip_ok_case(
                output_root, zips_dir, ml, mp, eager, b, sl, results_map, top_ops_by_case
            )
        _merge_top_ops_from_disk(output_root, top_ops_by_case)
        final_results = _ordered_case_results(output_root, cases, results_map)
        _write_summary(output_root, final_results, top_ops_by_case)
        return

    results_map = _load_results_map(output_root)
    top_ops_by_case: dict[str, list[dict[str, Any]]] = {}

    def persist() -> None:
        ordered = _ordered_case_results(output_root, cases, results_map)
        (output_root / RESULTS_JSON).write_text(
            json.dumps([asdict(x) for x in ordered], indent=2),
            encoding="utf-8",
        )

    for ml, mp, eager, b, sl in cases:
        case_id = _case_dir_name(ml, eager, b, sl)
        profile_dir = output_root / case_id
        zip_path = zips_dir / f"{case_id}_ascend_profile.zip"
        marker = profile_dir / MARKER_OK

        if args.skip_ok and _apply_skip_ok_case(
            output_root, zips_dir, ml, mp, eager, b, sl, results_map, top_ops_by_case
        ):
            persist()
            print(f"\n=== {case_id} (skip existing) ===", flush=True)
            continue

        profile_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--model",
            mp,
            "--batch-size",
            str(b),
            "--seq-len",
            str(sl),
            "--profile-dir",
            str(profile_dir),
            "--enforce-eager",
        ]

        print(f"\n=== {case_id} ===", flush=True)
        print(" ".join(cmd), flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        dt = time.perf_counter() - t0
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")

        err_snip = None
        status = "ok"
        if proc.returncode != 0:
            if _is_oom(out):
                status = "oom"
            else:
                status = "error"
            err_snip = out.strip()[-2000:]

        print(f"finish_time_sec={dt:.3f} status={status} returncode={proc.returncode}", flush=True)

        zip_saved = None
        op_csv = _find_op_statistic(profile_dir)

        if status == "ok" and op_csv:
            top_ops_by_case[case_id] = _read_top_ops(op_csv)
            try:
                _zip_profile_tree(profile_dir, zip_path)
                zip_saved = str(zip_path)
                marker.write_text(
                    json.dumps(
                        {"case_id": case_id, "duration_sec": dt, "zip": zip_saved},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as e:
                err_snip = str(e)
                status = "error"

        elif status == "ok" and not op_csv:
            status = "error"
            err_snip = "ok exit but op_statistic.csv not found"

        results_map[case_id] = CaseResult(
            case_id=case_id,
            model_label=ml,
            model_path=mp,
            eager=eager,
            batch_size=b,
            seq_len=sl,
            profile_dir=str(profile_dir),
            status=status,
            duration_sec=dt,
            error_snippet=err_snip,
            zip_path=zip_saved,
        )

        persist()

    _merge_top_ops_from_disk(output_root, top_ops_by_case)
    final_results = _ordered_case_results(output_root, cases, results_map)
    _write_summary(output_root, final_results, top_ops_by_case)


if __name__ == "__main__":
    main()
