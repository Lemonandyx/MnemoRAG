#!/usr/bin/env python3
"""Plot the retrieval-only 1x3 nprobe comparison (PNG and PDF)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LABELS = {
    "mnemorag": "MnemoRAG",
    "faiss": "Faiss",
    "proximity": "Proximity",
    "qvcache": "QVCache",
    "has": "HaS (shared-index)",
}
STYLES = {
    "mnemorag": ("#F57C20", "D"),
    "faiss": ("#00815D", "o"),
    "proximity": ("#3B6FB6", "s"),
    "qvcache": ("#C44E52", "^"),
    "has": ("#C69214", "v"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.results.glob("*/summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "complete" and row.get("system") in LABELS:
            rows.append(row)
    expected = 5 * 3 * 7
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} complete points, found {len(rows)}")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#4D4D4D",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.15), sharey=True)
    for panel, (axis, nprobe) in enumerate(zip(axes, (128, 256, 512))):
        for system in LABELS:
            selected = sorted(
                (r for r in rows if r["system"] == system and r["nprobe"] == nprobe),
                key=lambda r: r["arrival_rate_rps"],
            )
            color, marker = STYLES[system]
            axis.plot(
                [r["arrival_rate_rps"] for r in selected],
                [r["latency_mean_s"] for r in selected],
                label=LABELS[system],
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.5,
            )
        axis.set_title(f"({chr(97 + panel)}) nprobe={nprobe}", pad=7)
        axis.set_xlabel("Request Rate (req/s)", labelpad=2)
        axis.set_xticks((4, 8, 12, 16, 20, 24, 28))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="y", linestyle="--", linewidth=0.7, color="#C7C7C7", alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Retrieval Latency (s)", labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 1.015),
        columnspacing=1.35,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.77, wspace=0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
