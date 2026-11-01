#!/usr/bin/env python3
"""Plot a publication-ready heatmap of top-100 document access hotness."""

from __future__ import annotations

import argparse
import csv
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


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(DATASETS), 100), np.nan, dtype=float)
    raw_counts = np.zeros((len(DATASETS), 100), dtype=int)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset_index = DATASETS.index(row["dataset"])
            rank_index = int(row["document_rank"]) - 1
            matrix[dataset_index, rank_index] = float(row["normalized_hotness"])
            raw_counts[dataset_index, rank_index] = int(row["access_count"])
    if np.isnan(matrix).any():
        raise ValueError("The source CSV does not contain all 4 x 100 heatmap cells")
    return matrix, raw_counts


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


def smooth_rank_axis(
    matrix: np.ndarray,
    sigma: float,
    output_points: int = 800,
) -> np.ndarray:
    """Smooth only along document rank, then upsample for continuous rendering."""
    if sigma < 0:
        raise ValueError("Smoothing sigma must be non-negative")
    smoothed = matrix.copy()
    if sigma > 0:
        radius = max(1, int(np.ceil(4 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(matrix, ((0, 0), (radius, radius)), mode="edge")
        smoothed = np.vstack([np.convolve(row, kernel, mode="valid") for row in padded])
        # Ranked access frequency should remain non-increasing after smoothing.
        smoothed = np.minimum.accumulate(smoothed, axis=1)
        smoothed /= smoothed[:, :1]

    source_x = np.linspace(0.0, 100.0, matrix.shape[1])
    target_x = np.linspace(0.0, 100.0, output_points)
    return np.vstack([np.interp(target_x, source_x, row) for row in smoothed])


def plot_heatmap(
    source_csv: Path,
    output_base: Path,
    smoothing_sigma: float = 2.0,
) -> list[Path]:
    matrix, _ = load_matrix(source_csv)
    display_matrix = smooth_rank_axis(matrix, smoothing_sigma)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        interpolation="bilinear",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 100.0, len(DATASETS), 0.0),
    )

    ax.set_yticks(
        np.arange(len(DATASETS)) + 0.5,
        [DISPLAY_NAMES[dataset] for dataset in DATASETS],
    )
    ax.set_xticks(
        [0, 25, 50, 75, 100],
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
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    args = parser.parse_args()
    for output in plot_heatmap(
        args.source_csv,
        args.output_base,
        smoothing_sigma=args.smoothing_sigma,
    ):
        print(output.resolve())


if __name__ == "__main__":
    main()
