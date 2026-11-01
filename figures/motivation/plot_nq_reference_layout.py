#!/usr/bin/env python3
"""Plot the NQ background characterization in the requested two-panel layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_all(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_base.with_suffix(suffix),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.linewidth": 1.0,
            "legend.frameon": True,
            "legend.fancybox": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    stage_rows = read_csv(args.input_dir / "figure2a_nq_summary.csv")
    stage_rows.sort(key=lambda row: int(row["topk"]))
    timing_rows = [
        row
        for row in read_csv(args.input_dir / "figure2b_retrieval_latency.csv")
        if row["dataset"] == "nq"
    ]
    recall_rows = [
        row
        for row in read_csv(args.input_dir / "figure2c_recall_at_100.csv")
        if row["dataset"] == "nq"
    ]
    recall_rows.sort(key=lambda row: int(row["nprobe"]))

    fig, (bar_ax, latency_ax) = plt.subplots(1, 2, figsize=(10.0, 3.65))

    # (a) Stage latency.
    x = np.arange(len(stage_rows))
    width = 0.22
    bar_series = (
        ("mean_retrieval_ms", "Retrieval", "#F29A61", "x"),
        ("mean_prefill_ttft_ms", "Prefill", "#A8D5AE", "/"),
        ("mean_decode_ms", "Decode", "#88A9C5", "\\"),
    )
    for series_index, (key, label, color, hatch) in enumerate(bar_series):
        values = np.array([float(row[key]) for row in stage_rows])
        bar_ax.bar(
            x + (series_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#4A4A4A",
            linewidth=0.7,
            hatch=hatch,
        )
    bar_ax.set_xticks(x, [row["topk"] for row in stage_rows])
    bar_ax.set_xlabel("Top-k")
    bar_ax.set_ylabel("Latency (ms)")
    bar_ax.set_ylim(bottom=0)
    bar_ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.45)
    bar_ax.set_title("(a) Retrieval vs Prefill vs Decode", fontweight="bold", pad=6)
    bar_ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=9,
        borderpad=0.35,
        columnspacing=1.2,
    )

    # (b) Retrieval latency and Recall@100.  Use categorical positions to
    # reproduce the reference figure's evenly spaced powers-of-two ticks.
    probes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    positions = np.arange(len(probes))
    nprobe_map = {
        int(row["value"]): float(row["mean_latency_ms"])
        for row in timing_rows
        if row["parameter"] == "nprobe"
    }
    rbs_map = {
        int(row["value"]): float(row["mean_latency_ms"])
        for row in timing_rows
        if row["parameter"] == "rbs"
    }
    recall_map = {
        int(row["nprobe"]): float(row["mean_recall_at_100"]) for row in recall_rows
    }
    nprobe_line = latency_ax.plot(
        positions,
        [nprobe_map[value] for value in probes],
        color="#FF7F0E",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="nprobe latency",
    )[0]
    rbs_line = latency_ax.plot(
        positions,
        [rbs_map[value] for value in probes],
        color="#00834E",
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        label="RBS latency",
    )[0]
    latency_ax.set_xticks(positions, [str(value) for value in probes])
    latency_ax.set_xlabel("nprobe / RBS")
    latency_ax.set_ylabel("Retrieval Latency (ms)")
    latency_ax.set_yscale("log", base=10)
    latency_ax.set_ylim(8, 30000)
    latency_ax.set_yticks([10, 100, 1000, 10000])
    latency_ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    latency_ax.yaxis.set_minor_locator(mticker.NullLocator())
    latency_ax.grid(axis="y", which="major", linestyle="--", linewidth=0.65, alpha=0.45)
    latency_ax.set_title("(b) Retrieval Latency & Recall@100", fontweight="bold", pad=6)

    recall_ax = latency_ax.twinx()
    recall_line = recall_ax.plot(
        positions,
        [recall_map[value] for value in probes],
        color="#3775BA",
        marker="^",
        markersize=4.8,
        linewidth=1.8,
        linestyle="--",
        label="Recall@100 (k=100)",
    )[0]
    recall_ax.set_ylabel("Recall@100")
    recall_ax.set_ylim(0, 1.02)
    recall_ax.set_yticks(np.linspace(0, 1, 6))
    latency_ax.legend(
        [nprobe_line, rbs_line, recall_line],
        ["nprobe", "RBS", "Recall@100"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=9,
        borderpad=0.35,
        columnspacing=1.2,
    )

    for axis in (bar_ax, latency_ax, recall_ax):
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#555555")
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.18, top=0.77, wspace=0.18)
    save_all(fig, args.output_base)
    plt.close(fig)


if __name__ == "__main__":
    main()
