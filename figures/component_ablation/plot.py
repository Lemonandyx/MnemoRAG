#!/usr/bin/env python3
"""Plot component speedup and quality from the aggregate table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"variant", "speedup", "f1", "accuracy", "recall"}


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("the input table is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for row in rows:
        for field in REQUIRED_COLUMNS - {"variant"}:
            float(row[field])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Component-ablation aggregate CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/component_ablation",
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows)
    labels = [row["variant"] for row in rows]
    x = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#4D4D4D",
            "pdf.fonttype": 42,
        }
    )
    fig, (speed_axis, quality_axis) = plt.subplots(1, 2, figsize=(9.2, 3.35))
    speed = [float(row["speedup"]) for row in rows]
    bars = speed_axis.bar(
        x,
        speed,
        color="#F3A04A",
        edgecolor="#4D4D4D",
        linewidth=0.9,
        hatch="/",
        alpha=0.9,
    )
    speed_axis.axhline(1.0, color="#707070", linestyle="--", linewidth=0.9)
    speed_axis.bar_label(
        bars,
        labels=[f"{value:.2f}x" for value in speed],
        padding=3,
    )
    speed_axis.set_xticks(x, labels)
    speed_axis.set_ylabel("Speedup")
    speed_axis.set_title("Latency", fontweight="bold")
    speed_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    width = 0.24
    series = (
        ("f1", "F1", "#F2B66D", "x"),
        ("accuracy", "Accuracy", "#A8CEA8", "/"),
        ("recall", "Recall", "#86A8C5", ""),
    )
    for index, (key, label, color, hatch) in enumerate(series):
        quality_axis.bar(
            x + (index - 1) * width,
            [float(row[key]) for row in rows],
            width,
            label=label,
            color=color,
            edgecolor="#4D4D4D",
            linewidth=0.8,
            hatch=hatch,
            alpha=0.9,
        )
    quality_axis.set_xticks(x, labels)
    quality_axis.set_ylabel("Score")
    quality_axis.set_ylim(bottom=0)
    quality_axis.set_title("Quality", fontweight="bold")
    quality_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    quality_axis.legend(loc="upper center", ncol=3, frameon=True)

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.17,
        top=0.84,
        wspace=0.18,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
