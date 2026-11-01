#!/usr/bin/env python3
"""Create publication-ready reproductions of MnemoRAG Figures 2, 4, and 5."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("nq", "hotpotqa", "squad", "2wikimultihopqa")
DISPLAY_NAMES = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "squad": "SQuAD",
    "2wikimultihopqa": "2WikiMultiHopQA",
}
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 11
    axes_linewidth: float = 1.4


def apply_publication_style(style: FigureStyle) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_figure2(input_dir: Path, output_dir: Path) -> list[Path]:
    timing_rows = read_rows(input_dir / "figure2b_retrieval_latency.csv")
    recall_rows = read_rows(input_dir / "figure2c_recall_at_100.csv")
    by_dataset_timing = {
        dataset: [row for row in timing_rows if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 13.0), squeeze=False)
    stage_colors = (
        PALETTE["blue_main"],
        PALETTE["green_3"],
        PALETTE["red_strong"],
    )
    stage_hatches = ("", "//", "..")
    stages = (
        ("mean_retrieval_ms", "stdev_retrieval_ms", "Retrieval"),
        ("mean_prefill_ttft_ms", "stdev_prefill_ttft_ms", "Prefill"),
        ("mean_decode_ms", "stdev_decode_ms", "Decode"),
    )
    first_recall_ax = None

    for index, dataset in enumerate(DATASETS):
        ax = axes[index, 0]
        summary = read_rows(input_dir / f"figure2a_{dataset}_summary.csv")
        summary.sort(key=lambda row: int(row["topk"]))
        x = np.arange(len(summary))
        width = 0.24
        for stage_index, (mean_key, std_key, label) in enumerate(stages):
            values = np.array([float(row[mean_key]) / 1000 for row in summary])
            errors = np.array([float(row[std_key]) / 1000 for row in summary])
            ax.bar(
                x + (stage_index - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=stage_colors[stage_index],
                edgecolor="black",
                linewidth=0.7,
                hatch=stage_hatches[stage_index],
                label=label,
            )
        ax.set_xticks(x, [row["topk"] for row in summary])
        ax.set_xlabel("Retrieved documents (top-k)")
        ax.set_ylabel("Mean latency (s)")
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"({chr(97 + index * 2)}) {DISPLAY_NAMES[dataset]} — stage latency"
        )
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        ax = axes[index, 1]
        dataset_rows = by_dataset_timing[dataset]
        for parameter, label, color, marker in (
            ("nprobe", "Vary nprobe (batch=1)", PALETTE["blue_main"], "o"),
            ("rbs", "Vary RBS (nprobe=128)", PALETTE["red_strong"], "s"),
        ):
            rows = [row for row in dataset_rows if row["parameter"] == parameter]
            rows.sort(key=lambda row: int(row["value"]))
            values = np.array([int(row["value"]) for row in rows])
            means = np.array([float(row["mean_latency_ms"]) for row in rows])
            stdev = np.array([float(row["stdev_latency_ms"]) for row in rows])
            ax.plot(
                values,
                means,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.9,
                label=label,
            )
            ax.fill_between(
                values,
                np.maximum(means - stdev, 1e-3),
                means + stdev,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64, 256, 512], ["1", "4", "16", "64", "256", "512"])
        ax.set_xlabel("nprobe / retrieval batch size")
        ax.set_ylabel("Retrieval latency (ms, log scale)")
        ax.set_title(
            f"({chr(98 + index * 2)}) {DISPLAY_NAMES[dataset]} — retrieval scaling"
        )
        ax.grid(which="both", linestyle="--", linewidth=0.6, alpha=0.35)

        recall_ax = ax.twinx()
        if first_recall_ax is None:
            first_recall_ax = recall_ax
        dataset_recall = [row for row in recall_rows if row["dataset"] == dataset]
        dataset_recall.sort(key=lambda row: int(row["nprobe"]))
        recall_x = np.array([int(row["nprobe"]) for row in dataset_recall])
        recall_mean = np.array(
            [float(row["mean_recall_at_100"]) for row in dataset_recall]
        )
        recall_ax.plot(
            recall_x,
            recall_mean,
            color=PALETTE["teal"],
            marker="^",
            markersize=4,
            linewidth=1.9,
            linestyle="--",
            label="Recall@100 (k=100)",
        )
        recall_ax.set_ylim(0, 1.02)
        recall_ax.set_ylabel("Recall@100")
        recall_ax.spines["right"].set_visible(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    recall_handles, recall_labels = first_recall_ax.get_legend_handles_labels()
    fig.legend(
        handles + handles2 + recall_handles,
        labels + labels2 + recall_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.002),
        ncol=6,
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.0, w_pad=1.5)
    return finalize_figure(fig, output_dir / "figure2_background_reproduction")


def plot_figure4(input_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(input_dir / "figure4_semantic_similarity.csv")
    by_dataset = {row["dataset"]: row for row in rows}
    categories = [DISPLAY_NAMES[dataset] for dataset in DATASETS]
    series = (
        ("mean_inter_request_similarity", "Mean", "#F47C2C", "xx"),
        ("top1_retrieval_similarity", "Top1", "#237F4B", ""),
        ("top5_retrieval_similarity", "Top5", "#68A68A", ""),
        ("top10_retrieval_similarity", "Top10", "#A2C89F", ""),
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.35))
    x = np.arange(len(DATASETS))
    width = 0.19
    for series_index, (key, label, color, hatch) in enumerate(series):
        values = [float(by_dataset[dataset][key]) for dataset in DATASETS]
        ax.bar(
            x + (series_index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )
    ax.set_xticks(x, categories)
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("Cosine Similarity Score", fontsize=14)
    ax.set_ylim(0.72, 0.96)
    ax.set_title(
        "Semantic Similarity: Inter-Request vs. Top-k Document",
        fontsize=16,
        fontweight="bold",
        pad=6,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=4,
        fontsize=11,
        frameon=True,
        columnspacing=1.6,
        handletextpad=0.5,
        borderpad=0.3,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.73)
    return finalize_figure(fig, output_dir / "figure4_motivation_similarity")


def plot_figure5(input_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(input_dir / "figure5_document_hotness.csv")
    matrix = np.zeros((len(DATASETS), 100), dtype=float)
    for row in rows:
        dataset_index = DATASETS.index(row["dataset"])
        rank_index = int(row["document_rank"]) - 1
        matrix[dataset_index, rank_index] = float(row["normalized_hotness"])

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        extent=(0, 100, len(DATASETS), 0),
    )
    ax.set_yticks(
        np.arange(len(DATASETS)) + 0.5,
        [DISPLAY_NAMES[dataset] for dataset in DATASETS],
    )
    ax.set_xticks(
        [0, 25, 50, 75, 100],
        ["0%", "25%", "50%", "75%", "100%"],
    )
    ax.set_xlabel("Document Percentage", fontsize=13)
    ax.set_title("Document Access Hotness", fontsize=16, fontweight="bold", pad=6)
    ax.tick_params(axis="both", labelsize=12, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#333333")

    colorbar = fig.colorbar(image, ax=ax, pad=0.045, fraction=0.045, aspect=22)
    colorbar.set_label(
        "Document Access Hotness", fontsize=12, rotation=270, labelpad=17
    )
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(labelsize=11, length=3)
    colorbar.outline.set_linewidth(0.8)
    fig.subplots_adjust(left=0.20, right=0.88, bottom=0.17, top=0.88)
    return finalize_figure(fig, output_dir / "figure5_motivation_document_hotness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    apply_publication_style(FigureStyle())
    outputs = []
    outputs.extend(plot_figure2(args.input_dir, args.output_dir))
    outputs.extend(plot_figure4(args.input_dir, args.output_dir))
    outputs.extend(plot_figure5(args.input_dir, args.output_dir))
    manifest = {
        "source_data": str(args.input_dir.resolve()),
        "outputs": [str(path.resolve()) for path in outputs],
        "datasets": list(DATASETS),
        "formats": ["png", "pdf"],
        "dpi": 300,
    }
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
