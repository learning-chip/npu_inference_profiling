#!/usr/bin/env python3
"""Plot prefill bench: TTFT, input TPS, and TTFT speedup vs Triton baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo-relative default data dir
BENCH_DIR = Path(__file__).resolve().parent / "bench_prefill_20260424"
OUT_DIR = Path(__file__).resolve().parent / "figure"

# Plot order / legend: user-specified case → color
CASE_STYLE = {
    "pto_mega": {"color": "#1f77b4", "label": "PTO mega"},
    "pto": {"color": "#ff7f0e", "label": "PTO"},
    "triton": {"color": "#2ca02c", "label": "Triton"},
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


def _plot_model(model_dir: Path, model_name: str, out_path: Path) -> None:
    cases = ["triton", "pto", "pto_mega"]
    data: dict[str, dict[int, dict]] = {}
    for c in cases:
        p = model_dir / f"{c}.jsonl"
        data[c] = _by_seq_len(_load_jsonl(p))

    seq_lens = sorted(data["triton"].keys())
    for c in cases:
        if set(data[c].keys()) != set(seq_lens):
            raise ValueError(
                f"{model_dir.name}/{c}.jsonl seq_len set mismatch vs triton: "
                f"{sorted(data[c].keys())} vs {seq_lens}"
            )

    baseline = data["triton"]
    # TTFT speedup vs triton: higher = faster first token (triton time / case time)
    speedup: dict[str, list[float]] = {c: [] for c in cases}
    for sl in seq_lens:
        t0 = float(baseline[sl]["median_ttft_ms"])
        for c in cases:
            t = float(data[c][sl]["median_ttft_ms"])
            speedup[c].append(t0 / t if t > 0 else float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True, constrained_layout=True)
    ax_ttft, ax_tps, ax_sp = axes

    for c in ("pto_mega", "pto", "triton"):
        sty = CASE_STYLE[c]
        xs = np.array(seq_lens, dtype=float)
        ttft = np.array([float(data[c][sl]["median_ttft_ms"]) for sl in seq_lens])
        tps = np.array([float(data[c][sl]["input_tps"]) for sl in seq_lens])
        sp = np.array(speedup[c])
        ax_ttft.plot(xs, ttft, marker="o", markersize=5, linewidth=2, label=sty["label"], color=sty["color"])
        ax_tps.plot(xs, tps, marker="o", markersize=5, linewidth=2, label=sty["label"], color=sty["color"])
        ax_sp.plot(xs, sp, marker="o", markersize=5, linewidth=2, label=sty["label"], color=sty["color"])

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):d}"))
        ax.grid(True, which="major", alpha=0.35)
        ax.legend(loc="best", fontsize=9)

    ax_ttft.set_ylabel("Median TTFT (ms)")
    ax_ttft.set_title(f"Prefill benchmark — {model_name} model")

    ax_tps.set_ylabel("Input throughput (tok/s)")

    ax_sp.set_ylabel("TTFT speedup vs Triton\n(triton TTFT / case TTFT)")
    ax_sp.axhline(1.0, color="#666666", linestyle="--", linewidth=1, zorder=0)

    ax_sp.set_xlabel("Sequence length (tokens)")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub, title in (("2B", "2B"), ("0.8B", "0.8B")):
        _plot_model(
            BENCH_DIR / sub,
            title,
            OUT_DIR / f"prefill_speedup_{sub.replace('.', '_')}.png",
        )


if __name__ == "__main__":
    main()
