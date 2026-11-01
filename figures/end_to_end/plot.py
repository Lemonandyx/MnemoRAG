#!/usr/bin/env python3
"""Plot the 3x4 end-to-end latency grid from the aggregate table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("nq", "hotpotqa", "squad", "2wikimultihopqa")
WORKFLOWS = ("onepass", "recomp", "irg")
DISPLAY = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "squad": "SQuAD",
    "2wikimultihopqa": "2Wiki",
    "onepass": "One-pass",
    "recomp": "RECOMP",
    "irg": "IRG",
    "flashrag": "FlashRAG",
    "hedrarag": "HedraRAG",
    "mnemorag": "MnemoRAG",
}
STYLES = {
    "flashrag": ("#4CA267", "^"),
    "hedrarag": ("#E77C43", "d"),
    "mnemorag": ("#3F6FAE", "o"),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row["headline_eligible"].lower() == "true"
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="End-to-end aggregate CSV produced by the benchmark campaign.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/end_to_end",
    )
    parser.add_argument("--slo-seconds", type=float, default=10.0)
    args = parser.parse_args()

    rows = load_rows(args.input)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#4D4D4D",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 7.2), sharey=True)
    for row_index, workflow in enumerate(WORKFLOWS):
        for column_index, dataset in enumerate(DATASETS):
            axis = axes[row_index, column_index]
            for system, (color, marker) in STYLES.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["workflow"] == workflow
                        and row["dataset"] == dataset
                        and row["system"] == system
                    ),
                    key=lambda row: float(row["qps"]),
                )
                if not selected:
                    continue
                x = [float(row["qps"]) for row in selected]
                raw_y = [float(row["mean_latency_s"]) for row in selected]
                y = [min(value, args.slo_seconds) for value in raw_y]
                axis.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=3.8,
                    linewidth=1.4,
                    label=DISPLAY[system],
                )
            axis.axhline(
                args.slo_seconds,
                color="#7A7A7A",
                linestyle="--",
                linewidth=0.8,
            )
            axis.set_ylim(0, args.slo_seconds)
            axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
            axis.set_title(
                f"{DISPLAY[dataset]} · {DISPLAY[workflow]}",
                fontsize=10,
                fontweight="bold",
            )
            if row_index == len(WORKFLOWS) - 1:
                axis.set_xlabel("Request Rate (req/s)")
            if column_index == 0:
                axis.set_ylabel("Latency (s/query)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.08,
        top=0.91,
        wspace=0.12,
        hspace=0.30,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
