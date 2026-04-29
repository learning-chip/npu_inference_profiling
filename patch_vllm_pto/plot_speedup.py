#!/usr/bin/env python3
"""Plot prefill bench: TTFT, input TPS, and TTFT speedup vs Triton baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
_ROOT = Path(__file__).resolve().parent

CASE_STYLE = {
    "pto_mega": {"color": "#1f77b4", "label": "PTO mega"},
    "pto": {"color": "#ff7f0e", "label": "PTO"},
    "triton": {"color": "k", "linestyle": "--", "label": "Triton"},
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _by_seq_len(rows: list[dict]) -> dict[int, dict]:
    return {int(r["seq_len"]): r for r in rows}


def _sanitize_filename(label: str) -> str:
    return label.replace(".", "_")


def _plot_model(model_dir: Path, model_name: str, out_path: Path) -> None:
    cases = ["triton", "pto", "pto_mega"]
    data: dict[str, dict[int, dict]] = {}
    for c in cases:
        p = model_dir / f"{c}.jsonl"
        data[c] = _by_seq_len(_load_jsonl(p))

    baseline_k = sorted(data["triton"].keys())
    union_k = sorted(set().union(*(set(data[c].keys()) for c in cases)))

    seq_sp = baseline_k

    baseline = data["triton"]
    speedup_k: dict[str, list[float]] = {c: [] for c in cases}
    for sl in seq_sp:
        t0 = float(baseline[sl]["median_ttft_ms"])
        for c in cases:
            if sl not in data[c]:
                raise ValueError(f"{model_dir.name}/{c}.jsonl missing seq_len={sl} (need for speedup)")
            t = float(data[c][sl]["median_ttft_ms"])
            speedup_k[c].append(t0 / t if t > 0 else float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True, constrained_layout=True)
    ax_sp, ax_ttft, ax_tps = axes

    extra_after_triton = bool(baseline_k) and max(union_k) > max(baseline_k)

    for c in ("pto_mega", "pto", "triton"):
        sty = CASE_STYLE[c]
        xs_c = np.array(sorted(data[c].keys()), dtype=float)
        ttft = np.array([float(data[c][int(sl)]["median_ttft_ms"]) for sl in xs_c])
        tps_a = np.array([float(data[c][int(sl)]["input_tps"]) for sl in xs_c])
        line_kw = dict(
            marker="o",
            markersize=5,
            linewidth=2,
            label=sty["label"],
            color=sty["color"],
            linestyle=sty.get("linestyle", "-"),
        )
        sp = np.array(speedup_k[c])
        ax_sp.plot(np.array(seq_sp, dtype=float), sp, **line_kw)
        ax_ttft.plot(xs_c, ttft, **line_kw)
        ax_tps.plot(xs_c, tps_a, **line_kw)

    if extra_after_triton and baseline_k:
        lo = float(max(baseline_k))
        hi = float(max(union_k))
        for ax in (ax_ttft, ax_tps):
            ax.axvspan(lo, hi, alpha=0.13, color="0.35", zorder=0)

    for ax in axes:
        ax.set_xscale("log", base=2)
        xt = union_k
        ax.set_xticks(xt)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):d}"))
        ax.grid(True, which="major", alpha=0.35)

    ax_sp.legend(loc="best", fontsize=9)
    ax_ttft.legend(loc="best", fontsize=9)
    ax_tps.legend(loc="best", fontsize=9)

    ax_sp.set_ylabel("TTFT speedup vs Triton\n(triton TTFT / case TTFT)")
    main_title = f"Prefill benchmark — {model_name} model"
    if extra_after_triton and baseline_k:
        last_b = int(max(baseline_k))
        last_u = int(max(union_k))
        subtitle = (
            f"(Triton baseline fails at seq_len={last_b})"
        )
        ax_sp.set_title(f"{main_title}\n{subtitle}", fontsize=10)
    else:
        ax_sp.set_title(main_title, fontsize=11)

    ax_ttft.set_ylabel("Median TTFT (ms)")

    ax_tps.set_ylabel("Input throughput (tok/s)")
    ax_tps.set_xlabel("Sequence length (tokens)")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TTFT/TPS/speedup from triton.jsonl / pto.jsonl / pto_mega.jsonl",
    )
    parser.add_argument(
        "--bench-dir",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "Root directory containing model subdirectories (each with *.jsonl). "
            "May be repeated. Default: two runs (20260424 small models + 20260428 GQA)."
        ),
    )
    parser.add_argument(
        "--bench-model",
        action="append",
        dest="bench_models",
        metavar="DIR:DISPLAY",
        help=(
            'Single dataset: bench root path and comma-separated MODEL:Title pairs '
            '(e.g. bench_prefill:"2B:2B,0.8B:0.8B"). Use with repeated --bench-model '
            "instead of defaults."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "figure",
        help="Output directory for PNGs (default: ./figure)",
    )
    args = parser.parse_args()

    if args.bench_models:
        plot_specs: list[tuple[Path, list[tuple[str, str]]]] = []
        for raw in args.bench_models:
            if ":" not in raw:
                raise SystemExit("--bench-model must contain ':' (DIR:S1,S2,...)")
            root_s, pairs_s = raw.split(":", 1)
            root = Path(root_s).expanduser()
            if not root.is_absolute():
                root = (_ROOT / root).resolve()
            pairs: list[tuple[str, str]] = []
            for part in pairs_s.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise SystemExit(f"bad MODEL:Title segment {part!r}")
                lab, tit = part.split(":", 1)
                pairs.append((lab.strip(), tit.strip() or lab.strip()))
            plot_specs.append((root, pairs))
    elif args.bench_dir:
        roots = []
        for p in args.bench_dir:
            r = Path(p).expanduser()
            roots.append(r if r.is_absolute() else (_ROOT / r).resolve())
        if len(roots) == 1:
            bench = roots[0]
            pairs = [(d.name, d.name) for d in sorted(bench.iterdir()) if d.is_dir()]
            plot_specs = [(bench, pairs)]
        else:
            plot_specs = []
            for bench in roots:
                subs = sorted(d.name for d in bench.iterdir() if d.is_dir())
                plot_specs.append((bench, [(s, s) for s in subs]))
    else:
        plot_specs = [
            (_ROOT / "bench_prefill_20260424", [("2B", "2B"), ("0.8B", "0.8B")]),
            (_ROOT / "bench_prefill_20260428_gqa", [("4B", "4B"), ("9B", "9B")]),
            (_ROOT / "bench_prefill_Qwen36_27B_w8a8", [("27B-w8a8", "27B-w8a8")]),
        ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bench_root, model_pairs in plot_specs:
        for subdir, title in model_pairs:
            model_dir = bench_root / subdir
            if not model_dir.is_dir():
                raise FileNotFoundError(f"missing benchmark dir {model_dir}")
            outp = args.out_dir / f"prefill_speedup_{_sanitize_filename(title)}.png"
            _plot_model(model_dir, title, outp)
            print(f"wrote {outp}")


if __name__ == "__main__":
    main()
