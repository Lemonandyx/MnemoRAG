#!/usr/bin/env python3
"""Plot a document-access heatmap from an external aggregate table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("the input table is empty")
    datasets = tuple(dict.fromkeys(row["dataset"] for row in rows))
    rank_count = max(int(row["document_rank"]) for row in rows)
    matrix = np.full((len(datasets), rank_count), np.nan, dtype=float)
    for row in rows:
        dataset_index = datasets.index(row["dataset"])
        rank_index = int(row["document_rank"]) - 1
        matrix[dataset_index, rank_index] = float(row["normalized_hotness"])
    if np.isnan(matrix).any():
        raise ValueError("the input table does not contain a complete matrix")
    return matrix, datasets


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_heatmap(
    source_csv: Path,
    output_base: Path,
) -> list[Path]:
    matrix, datasets = load_matrix(source_csv)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 1.0, len(datasets), 0.0),
    )

    ax.set_yticks(
        np.arange(len(datasets)) + 0.5,
        datasets,
    )
    ax.set_xticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["0%", "25%", "50%", "75%", "100%"],
    )
    ax.set_xlabel("Document Percentage", fontsize=12)
    ax.set_title("Document Access Hotness", fontsize=15, fontweight="bold", pad=4)
    ax.tick_params(axis="both", direction="out", length=3.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)

    colorbar = fig.colorbar(image, ax=ax, pad=0.05, fraction=0.04, aspect=22)
    colorbar.set_label("Document Access Hotness", rotation=270, labelpad=18)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(length=3, width=0.9)
    colorbar.outline.set_linewidth(0.9)

    fig.subplots_adjust(left=0.24, right=0.88, bottom=0.16, top=0.88)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        output = output_base.with_suffix(suffix)
        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
        outputs.append(output)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    for output in plot_heatmap(args.source_csv, args.output_base):
        print(output.resolve())


if __name__ == "__main__":
    main()
