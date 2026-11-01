#!/usr/bin/env python3
"""Summarize the formal campaign and render publication-ready figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .campaign_statistics import (
        aggregate_repeated_groups,
        mean_confidence_interval,
        robust_slo_pass,
    )
    from .run_full_e2e_campaign import atomic_write_json
    from .run_mnemo_ablation import VARIANTS
except ImportError:
    from campaign_statistics import (
        aggregate_repeated_groups,
        mean_confidence_interval,
        robust_slo_pass,
    )
    from run_full_e2e_campaign import atomic_write_json
    from run_mnemo_ablation import VARIANTS


SYSTEMS = ("mnemorag", "flashrag", "hedrarag")
DATASET_ORDER = ("squad", "nq", "hotpotqa", "2wikimultihopqa")
WORKFLOW_LABELS = {
    "onepass": "One-Pass [9]",
    "recomp": "RECOMP [50]",
    "irg": "IRG",
}
SYSTEM_LABELS = {
    "mnemorag": "MnemoRAG",
    "flashrag": "FlashRAG",
    "hedrarag": "HedraRAG*",
}
COLORS = {
    "mnemorag": "#0F4D92",
    "flashrag": "#8BCF8B",
    "hedrarag": "#E9A6A1",
}
MARKERS = {"mnemorag": "o", "flashrag": "s", "hedrarag": "^"}
HATCHES = {"mnemorag": "xx", "flashrag": "//", "hedrarag": "\\\\"}
ABLATION_LABELS = {
    "full": "Full",
    "no_sar": "w/o similarity-aware retrieval",
    "no_hotness": "w/o Heat",
    "no_async_pipeline": "w/o Pipeline",
    "no_cpu_kv": "w/o CPU-KV",
    "no_cache_aware_prefill": "w/o CA-Prefill",
    "no_history_warmstart": "Cold History",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-repeats", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def slo_capacities(aggregated: list[dict]) -> list[dict]:
    output = []
    datasets = sorted({str(row.get("dataset", "nq")) for row in aggregated})
    for dataset in datasets:
        for workflow in ("onepass", "recomp", "irg"):
            nprobes = sorted(
                {
                    int(row["nprobe"])
                    for row in aggregated
                    if row.get("dataset", "nq") == dataset
                    and row["workflow"] == workflow
                }
            )
            for nprobe in nprobes:
                for system in SYSTEMS:
                    rows = [
                        row
                        for row in aggregated
                        if row.get("dataset", "nq") == dataset
                        and row["workflow"] == workflow
                        and row["nprobe"] == nprobe
                        and row["system"] == system
                    ]
                    passing = [row for row in rows if robust_slo_pass(row)]
                    selected = (
                        max(passing, key=lambda row: row["arrival_rate_rps"])
                        if passing
                        else None
                    )
                    output.append(
                        {
                            "dataset": dataset,
                            "workflow": workflow,
                            "nprobe": nprobe,
                            "system": system,
                            "robust_slo_max_qps": (
                                float(selected["arrival_rate_rps"])
                                if selected is not None
                                else 0.0
                            ),
                            "has_passing_point": selected is not None,
                        }
                    )
    return output


def aggregate_ablation(
    summaries: list[dict], *, minimum_repeats: int = 2
) -> list[dict]:
    if minimum_repeats < 1:
        raise ValueError("minimum_repeats must be positive")
    output = []
    for variant in VARIANTS:
        rows = [row for row in summaries if row.get("campaign_variant") == variant]
        if len(rows) < minimum_repeats:
            raise ValueError(
                f"ablation {variant} has {len(rows)} repeats; "
                f"expected at least {minimum_repeats}"
            )
        output.append(
            {
                "variant": variant,
                "repeat_count": len(rows),
                "latency": {
                    name: mean_confidence_interval(
                        [float(row[name]) for row in rows],
                        allow_single=len(rows) == 1,
                    )
                    for name in (
                        "request_latency_mean_s",
                        "request_latency_p95_s",
                        "request_latency_p99_s",
                        "throughput_qps",
                    )
                },
                "quality": {
                    name: mean_confidence_interval(
                        [float(row["metrics"][name]) for row in rows],
                        allow_single=len(rows) == 1,
                    )
                    for name in ("retrieval_recall_top3", "f1", "acc")
                },
            }
        )
    return output


def matched_latency_speedups(
    aggregated: list[dict],
    *,
    metric: str,
) -> list[dict]:
    grouped = {}
    for row in aggregated:
        key = (
            row["dataset"],
            row["workflow"],
            row["nprobe"],
            row["arrival_rate_rps"],
        )
        grouped.setdefault(key, {})[row["system"]] = row
    output = []
    for key, systems in sorted(grouped.items()):
        if set(systems) != set(SYSTEMS):
            continue
        mnemo = systems["mnemorag"]["metrics"][metric]["mean"]
        for baseline in ("flashrag", "hedrarag"):
            baseline_latency = systems[baseline]["metrics"][metric]["mean"]
            output.append(
                {
                    "dataset": key[0],
                    "workflow": key[1],
                    "nprobe": key[2],
                    "arrival_rate_rps": key[3],
                    "baseline": baseline,
                    "metric": metric,
                    "mnemo_latency_s": mnemo,
                    "baseline_latency_s": baseline_latency,
                    "speedup": baseline_latency / mnemo,
                }
            )
    return output


def capacity_speedups(capacities: list[dict]) -> list[dict]:
    grouped = {}
    for row in capacities:
        grouped.setdefault((row["dataset"], row["workflow"], row["nprobe"]), {})[
            row["system"]
        ] = row
    output = []
    for (dataset, workflow, nprobe), systems in sorted(grouped.items()):
        mnemo = float(systems["mnemorag"]["robust_slo_max_qps"])
        for baseline in ("flashrag", "hedrarag"):
            reference = float(systems[baseline]["robust_slo_max_qps"])
            censored = bool(
                systems["mnemorag"].get("capacity_is_lower_bound", False)
                or systems[baseline].get("capacity_is_lower_bound", False)
            )
            output.append(
                {
                    "dataset": dataset,
                    "workflow": workflow,
                    "nprobe": nprobe,
                    "baseline": baseline,
                    "mnemo_max_qps": mnemo,
                    "baseline_max_qps": reference,
                    "capacity_speedup": (
                        mnemo / reference if reference > 0 and not censored else None
                    ),
                    "comparison_status": (
                        "censored_not_computed" if censored else "exact"
                    ),
                }
            )
    return output


def annotate_slo_censoring(
    capacities: list[dict],
    censoring: dict | None,
    workload_selection: dict,
) -> list[dict]:
    """Attach exact-boundary or right-censored-at-cap semantics to capacities."""
    metadata = censoring or {}
    crossings = metadata.get("crossings", {})
    cap_rows = {
        (
            str(row["dataset"]),
            str(row["workflow"]),
            int(row["nprobe"]),
            str(row["system"]),
        ): row
        for row in metadata.get("systems_reaching_qps_cap_without_violation", [])
    }
    configured_caps = metadata.get("qps_cap_by_workflow", {})
    if not configured_caps:
        configured_caps = workload_selection.get("candidate_slo_early_stop", {}).get(
            "maximum_qps_by_workflow", {}
        )
    for row in capacities:
        scan_key = "|".join(map(str, (row["dataset"], row["workflow"], row["nprobe"])))
        crossing = crossings.get(scan_key, {}).get(row["system"])
        cap_row = cap_rows.get(
            (
                str(row["dataset"]),
                str(row["workflow"]),
                int(row["nprobe"]),
                str(row["system"]),
            )
        )
        qps_cap = (
            float(cap_row["qps_cap"])
            if cap_row is not None
            else (
                float(configured_caps[row["workflow"]])
                if row["workflow"] in configured_caps
                else None
            )
        )
        if crossing is not None:
            boundary_status = "observed_first_violation"
            first_violating_qps = float(crossing["qps"])
            capacity_is_lower_bound = False
        elif cap_row is not None:
            boundary_status = "right_censored_at_qps_cap_without_violation"
            first_violating_qps = None
            capacity_is_lower_bound = True
        else:
            boundary_status = "censoring_metadata_unavailable"
            first_violating_qps = None
            capacity_is_lower_bound = False
        row.update(
            {
                "boundary_status": boundary_status,
                "first_violating_qps": first_violating_qps,
                "qps_scan_cap": qps_cap,
                "capacity_is_lower_bound": capacity_is_lower_bound,
                "extrapolated_beyond_qps_cap": False,
            }
        )
    return capacities


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flatten_aggregated(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        base = {
            name: row[name]
            for name in (
                "dataset",
                "workflow",
                "topk",
                "nprobe",
                "arrival_rate_rps",
                "system",
                "repeat_count",
            )
        }
        for group in ("metrics", "quality"):
            for metric, statistics in row[group].items():
                for statistic, value in statistics.items():
                    base[f"{group}.{metric}.{statistic}"] = value
        output.append(base)
    return output


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return plt


def plot_mean_latency_qps_grid(
    rows: list[dict], output_dir: Path, *, nprobe: int = 128
) -> None:
    plt = configure_matplotlib()
    workflows = ("onepass", "recomp", "irg")
    datasets = tuple(
        dataset
        for dataset in DATASET_ORDER
        if any(
            row["dataset"] == dataset and int(row["nprobe"]) == nprobe for row in rows
        )
    )
    if datasets != DATASET_ORDER:
        raise ValueError(
            f"mean-latency grid requires datasets {DATASET_ORDER}, observed {datasets}"
        )
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(13.6, 8.4),
        sharey=True,
        squeeze=False,
    )
    for row_index, workflow in enumerate(workflows):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            for system in SYSTEMS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["workflow"] == workflow
                        and row["nprobe"] == nprobe
                        and row["system"] == system
                    ),
                    key=lambda row: row["arrival_rate_rps"],
                )
                x = [row["arrival_rate_rps"] for row in selected]
                y = [
                    row["metrics"]["request_latency_mean_s"]["mean"] for row in selected
                ]
                low = [
                    row["metrics"]["request_latency_mean_s"]["ci_low"]
                    for row in selected
                ]
                high = [
                    row["metrics"]["request_latency_mean_s"]["ci_high"]
                    for row in selected
                ]
                axis.plot(
                    x,
                    y,
                    color=COLORS[system],
                    marker=MARKERS[system],
                    linewidth=2,
                    markersize=4.5,
                    label=SYSTEM_LABELS[system],
                )
                axis.fill_between(x, low, high, color=COLORS[system], alpha=0.14)
            axis.axhline(10.0, color="#4D4D4D", linestyle="--", linewidth=1.3)
            axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            if row_index == 0:
                axis.set_title(dataset, fontsize=10.5)
            if row_index == 2:
                axis.set_xlabel("Request rate (QPS)")
            if column_index == 0:
                axis.set_ylabel(f"{WORKFLOW_LABELS[workflow]}\nMean E2E latency (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=1.2)
    save_figure(fig, output_dir, "e2e_mean_latency_qps_3x4")
    plt.close(fig)


def plot_slo_capacity(rows: list[dict], output_dir: Path, *, dataset: str) -> None:
    plt = configure_matplotlib()
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=True)
    nprobes = tuple(
        sorted({int(row["nprobe"]) for row in rows if row["dataset"] == dataset})
    )
    x = np.arange(len(nprobes))
    width = 0.24
    for axis, workflow in zip(axes, ("onepass", "recomp", "irg")):
        for system_index, system in enumerate(SYSTEMS):
            values = [
                next(
                    row["robust_slo_max_qps"]
                    for row in rows
                    if row["workflow"] == workflow
                    and row["dataset"] == dataset
                    and row["nprobe"] == nprobe
                    and row["system"] == system
                )
                for nprobe in nprobes
            ]
            axis.bar(
                x + (system_index - 1) * width,
                values,
                width,
                color=COLORS[system],
                edgecolor="#4D4D4D",
                linewidth=0.8,
                alpha=0.9,
                hatch=HATCHES[system],
                label=SYSTEM_LABELS[system],
            )
        axis.set_xticks(x, tuple(map(str, nprobes)))
        axis.set_xlabel("nprobe")
        axis.set_title(WORKFLOW_LABELS[workflow])
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axes[0].set_ylabel("Max QPS satisfying 10 s SLO")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.91), pad=1.2)
    save_figure(fig, output_dir, f"robust_slo_capacity_{dataset}")
    plt.close(fig)


def plot_dataset_latency(rows: list[dict], output_dir: Path) -> None:
    plt = configure_matplotlib()
    import numpy as np

    datasets = ("squad", "nq", "hotpotqa", "2wikimultihopqa")
    labels = ("SQuAD", "NQ", "HotpotQA", "2Wiki")
    workflows = ("onepass", "recomp", "irg")
    nprobes = tuple(sorted({int(row["nprobe"]) for row in rows}))
    x = np.arange(len(datasets))
    width = 0.24
    fig, axes = plt.subplots(
        3,
        len(nprobes),
        figsize=(max(7.6, 3.7 * len(nprobes)), 8.4),
        sharex=True,
        squeeze=False,
    )
    for row_index, workflow in enumerate(workflows):
        for column_index, nprobe in enumerate(nprobes):
            axis = axes[row_index, column_index]
            for system_index, system in enumerate(SYSTEMS):
                selected = [
                    next(
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["workflow"] == workflow
                        and row["nprobe"] == nprobe
                        and row["system"] == system
                    )
                    for dataset in datasets
                ]
                means = [
                    row["metrics"]["request_latency_p95_s"]["mean"] for row in selected
                ]
                errors = [
                    row["metrics"]["request_latency_p95_s"]["ci_half_width"]
                    for row in selected
                ]
                axis.bar(
                    x + (system_index - 1) * width,
                    means,
                    width,
                    yerr=errors,
                    capsize=2.0,
                    color=COLORS[system],
                    edgecolor="#4D4D4D",
                    linewidth=0.8,
                    alpha=0.9,
                    hatch=HATCHES[system],
                    label=SYSTEM_LABELS[system],
                )
            axis.set_xticks(x, labels, rotation=18, ha="right")
            axis.set_title(
                f"{WORKFLOW_LABELS[workflow]}, nprobe={nprobe}", fontsize=10.5
            )
            axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            if column_index == 0:
                axis.set_ylabel("p95 E2E latency (s)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=1.2)
    save_figure(fig, output_dir, "multidataset_e2e_latency")
    plt.close(fig)


def plot_ablation(rows: list[dict], output_dir: Path) -> None:
    plt = configure_matplotlib()
    import numpy as np

    labels = tuple(ABLATION_LABELS[variant] for variant in VARIANTS)
    x = np.arange(len(rows))
    full_latency = rows[0]["latency"]["request_latency_mean_s"]["mean"]
    normalized = [
        row["latency"]["request_latency_mean_s"]["mean"] / full_latency for row in rows
    ]
    recall_delta = [
        100.0
        * (
            row["quality"]["retrieval_recall_top3"]["mean"]
            - rows[0]["quality"]["retrieval_recall_top3"]["mean"]
        )
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axes[0].bar(
        x,
        normalized,
        color="#3775BA",
        edgecolor="#4D4D4D",
        linewidth=0.8,
        alpha=0.9,
        hatch="//",
    )
    axes[0].axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Normalized mean E2E latency")
    axes[1].bar(
        x,
        recall_delta,
        color="#8BCF8B",
        edgecolor="#4D4D4D",
        linewidth=0.8,
        alpha=0.9,
        hatch="\\\\",
    )
    axes[1].axhline(0.0, color="#4D4D4D", linewidth=1.0)
    axes[1].set_ylabel("Recall@3 change (pp)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axes[0].set_title("Latency")
    axes[1].set_title("Retrieval quality")
    fig.tight_layout(pad=1.2)
    save_figure(fig, output_dir, "mnemorag_component_ablation")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.minimum_repeats < 1:
        raise ValueError("--minimum-repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_summaries = load_json(
        args.campaign_root / "final_slo" / "final-slo_results.json"
    )
    dataset_summaries = load_json(
        args.campaign_root / "dataset_latency" / "dataset-latency_results.json"
    )
    ablation_summaries = load_json(
        args.campaign_root / "ablation" / "ablation_results.json"
    )
    full_split_path = args.campaign_root / "mnemo_full" / "mnemo-full_results.json"
    full_split_summaries = (
        load_json(full_split_path) if full_split_path.exists() else []
    )
    candidate_summaries = load_json(
        args.campaign_root / "candidate" / "candidate_results.json"
    )
    final_aggregated = aggregate_repeated_groups(
        final_summaries,
        minimum_repeats=args.minimum_repeats,
        allow_incomplete_system_groups=True,
    )
    dataset_aggregated = aggregate_repeated_groups(
        dataset_summaries, minimum_repeats=args.minimum_repeats
    )
    capacities = slo_capacities(final_aggregated)
    final_latency_speedups = matched_latency_speedups(
        final_aggregated, metric="request_latency_p95_s"
    )
    dataset_latency_speedups = matched_latency_speedups(
        dataset_aggregated, metric="request_latency_p95_s"
    )
    ablation = aggregate_ablation(
        ablation_summaries, minimum_repeats=args.minimum_repeats
    )
    quality = {
        "final_slo": load_json(
            args.campaign_root / "analysis" / "final_slo_quality_noninferiority.json"
        ),
        "dataset_latency": load_json(
            args.campaign_root
            / "analysis"
            / "dataset_latency_quality_noninferiority.json"
        ),
    }
    hedra_fidelity = sorted(
        {
            json.dumps(row.get("implementation_fidelity"), sort_keys=True)
            for row in final_summaries
            if row["system"] == "hedrarag"
        }
    )
    full_split_disclosure = [
        {
            "dataset": row["dataset"],
            "workflow": row["workflow"],
            "nprobe": row["nprobe"],
            "arrival_rate_rps": row["arrival_rate_rps"],
            "num_requests": row["num_requests"],
            "split_disclosure": row["split_disclosure"],
            "slo_attainment": row["slo_attainment"],
            "slo_goodput_qps": row["slo_goodput_qps"],
            "request_latency_p50_s": row["request_latency_p50_s"],
            "request_latency_p95_s": row["request_latency_p95_s"],
            "request_latency_p99_s": row["request_latency_p99_s"],
            "provenance": row["provenance"],
        }
        for row in full_split_summaries
    ]
    legacy_counts = {
        system: sum(
            (row.get("provenance") or {}).get("classification") == "legacy_semantics"
            for row in candidate_summaries
            if row["system"] == system
        )
        for system in SYSTEMS
    }
    sample_disclosure = {
        "matched_final_slo": {
            system: sorted(
                {
                    int(row["num_requests"])
                    for row in final_summaries
                    if row["system"] == system
                }
            )
            for system in SYSTEMS
        },
        "dataset_latency": {
            system: sorted(
                {
                    int(row["num_requests"])
                    for row in dataset_summaries
                    if row["system"] == system
                }
            )
            for system in SYSTEMS
        },
        "mnemorag_full_split": sorted(
            {(row["dataset"], int(row["num_requests"])) for row in full_split_summaries}
        ),
    }
    workload_selection_path = args.campaign_root / "workload_selection.json"
    workload_selection = (
        load_json(workload_selection_path) if workload_selection_path.exists() else {}
    )
    sample_disclosure["proxy_workload"] = workload_selection
    censoring_path = args.campaign_root / "final_slo" / "candidate_slo_censoring.json"
    slo_censoring = load_json(censoring_path) if censoring_path.exists() else None
    sample_disclosure["slo_censoring"] = slo_censoring
    capacities = annotate_slo_censoring(
        capacities,
        slo_censoring,
        workload_selection,
    )
    capacity_comparisons = capacity_speedups(capacities)
    report = {
        "status": "complete",
        "measurement_repeat_count": args.minimum_repeats,
        "repeat_statistics": {
            "cross_run_confidence_intervals_available": args.minimum_repeats >= 2,
            "single_measurement_points": args.minimum_repeats == 1,
            "slo_decision_basis": (
                "single_measurement_point_estimate"
                if args.minimum_repeats == 1
                else "95_percent_repeat_level_ci"
            ),
            "quality_interval_basis": ("paired_request_level_bootstrap_512_queries"),
        },
        "evaluation_scope": workload_selection.get("evaluation_scope"),
        "offline_profile_observation": workload_selection.get("profile_evidence_role"),
        "quality_noninferiority_all_pass": all(
            point["passes"] for report_rows in quality.values() for point in report_rows
        ),
        "slo_capacities": capacities,
        "slo_capacity_speedups": capacity_comparisons,
        "sample_and_split_disclosure": sample_disclosure,
        "legacy_semantics_reuse_count": legacy_counts,
        "mnemorag_full_split_point_count": len(full_split_summaries),
        "worst_dataset_p95_speedup": {
            baseline: min(
                row["speedup"]
                for row in dataset_latency_speedups
                if row["baseline"] == baseline
            )
            for baseline in ("flashrag", "hedrarag")
        },
        "hedrarag_implementation_fidelity": [json.loads(row) for row in hedra_fidelity],
        "hedrarag_headline_eligible": False,
        "hedrarag_label_note": (
            "HedraRAG* uses the artifact compatibility backend and is not a native "
            "CUDA headline result when headline_eligible is false."
        ),
    }
    atomic_write_json(args.output_dir / "final_report.json", report)
    atomic_write_json(args.output_dir / "final_slo_aggregated.json", final_aggregated)
    atomic_write_json(
        args.output_dir / "dataset_latency_aggregated.json", dataset_aggregated
    )
    atomic_write_json(args.output_dir / "ablation_aggregated.json", ablation)
    atomic_write_json(
        args.output_dir / "mnemorag_full_split_results.json",
        full_split_disclosure,
    )
    atomic_write_json(
        args.output_dir / "final_slo_latency_speedups.json",
        final_latency_speedups,
    )
    atomic_write_json(
        args.output_dir / "dataset_latency_speedups.json",
        dataset_latency_speedups,
    )
    final_flat = flatten_aggregated(final_aggregated)
    dataset_flat = flatten_aggregated(dataset_aggregated)
    if final_flat:
        write_rows(
            args.output_dir / "final_slo_aggregated.csv",
            final_flat,
            list(final_flat[0]),
        )
    if dataset_flat:
        write_rows(
            args.output_dir / "dataset_latency_aggregated.csv",
            dataset_flat,
            list(dataset_flat[0]),
        )
    write_rows(
        args.output_dir / "slo_capacities.csv",
        capacities,
        [
            "dataset",
            "workflow",
            "nprobe",
            "system",
            "robust_slo_max_qps",
            "has_passing_point",
            "boundary_status",
            "first_violating_qps",
            "qps_scan_cap",
            "capacity_is_lower_bound",
            "extrapolated_beyond_qps_cap",
        ],
    )
    write_rows(
        args.output_dir / "slo_capacity_speedups.csv",
        capacity_comparisons,
        [
            "dataset",
            "workflow",
            "nprobe",
            "baseline",
            "mnemo_max_qps",
            "baseline_max_qps",
            "capacity_speedup",
            "comparison_status",
        ],
    )
    write_rows(
        args.output_dir / "dataset_latency_speedups.csv",
        dataset_latency_speedups,
        [
            "dataset",
            "workflow",
            "nprobe",
            "arrival_rate_rps",
            "baseline",
            "metric",
            "mnemo_latency_s",
            "baseline_latency_s",
            "speedup",
        ],
    )
    plot_mean_latency_qps_grid(final_aggregated, args.output_dir, nprobe=128)
    for dataset in sorted({row["dataset"] for row in final_aggregated}):
        plot_slo_capacity(capacities, args.output_dir, dataset=dataset)
    plot_dataset_latency(dataset_aggregated, args.output_dir)
    plot_ablation(ablation, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
