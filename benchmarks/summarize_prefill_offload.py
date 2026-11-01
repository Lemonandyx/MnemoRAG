#!/usr/bin/env python3
"""Aggregate prefill-trigger and CPU-KV transfer experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

POLICIES = ("sequential", "stage_aware", "prefill_batch", "prefill_start")
PERFORMANCE_KEYS = (
    "wall_time_s",
    "throughput_qps",
    "request_latency_mean_s",
    "request_latency_p95_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def load_point(
    path: Path,
    expected_policy: str,
    expected_replicates: int,
    trace_mode: str,
) -> list[dict]:
    rows = []
    for summary_path in sorted(path.glob(f"{expected_policy}_r*/**/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        topology = row.get("cpu_topology") or {}
        preflight = row.get("resource_preflight") or {}
        transfer = row.get("cpu_kv_transfer_latency") or {}
        d2h = transfer.get("d2h_offload") or {}
        runtime = row.get("runtime_resource_monitor") or {}
        if (
            row.get("status") != "complete"
            or row.get("async_retrieval_policy") != expected_policy
            or topology.get("physical_core_count") != 64
            or topology.get("smt_siblings_selected")
            or preflight.get("violations")
            or not row.get("cpu_kv_offload", {}).get("enabled")
            or int(d2h.get("worker_completed_jobs", 0)) <= 0
            or not row.get("native_prefill_admission_event")
            or row.get("request_cycle_size", 0) != (0 if trace_mode == "unique" else 80)
            or row.get("components")
            != ("full" if trace_mode == "unique" else "async_baseline")
            or int(row.get("max_tokens", -1)) != (32 if trace_mode == "unique" else 8)
            or int(row.get("topk", -1)) != 5
            or int(row.get("nprobe", -1)) != 128
            or bool(row.get("query_base_exact_lookup")) != (trace_mode == "unique")
            or runtime.get("status") != "clean"
            or not runtime.get("continuous")
            or int(runtime.get("sample_count", 0)) < 2
            or int(runtime.get("foreign_gpu_process_count_max", -1)) != 0
            or float(runtime.get("external_cpu_busy_percent_max", float("inf"))) > 5.0
        ):
            raise RuntimeError(f"unverified point: {summary_path}")
        row["summary_path"] = str(summary_path)
        rows.append(row)
    if len(rows) != expected_replicates:
        raise RuntimeError(
            f"expected {expected_replicates} {expected_policy} points, found {len(rows)}"
        )
    return rows


def mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def aggregate(rows: list[dict]) -> dict:
    scheduler_keys = (
        "mean_retrieval_prefill_coverage",
        "mean_retrieval_decode_coverage",
        "mean_retrieval_generation_coverage",
        "mean_uncovered_retrieval_s",
    )
    result = {
        "replicates": len(rows),
        "summary_paths": [row["summary_path"] for row in rows],
    }
    for key in PERFORMANCE_KEYS:
        result[f"{key}_mean"] = mean(rows, key)
    for key in scheduler_keys:
        result[f"{key}_mean"] = statistics.fmean(
            float(row["batch_stage_scheduler"].get(key, 0.0)) for row in rows
        )
    for direction in ("h2d_reload", "d2h_offload"):
        result[f"{direction}_jobs"] = sum(
            int(row["cpu_kv_transfer_latency"][direction]["worker_completed_jobs"])
            for row in rows
        )
        result[f"{direction}_worker_total_s"] = sum(
            float(row["cpu_kv_transfer_latency"][direction]["worker_observed_total_s"])
            for row in rows
        )
    result["runtime_sample_count"] = sum(
        int(row["runtime_resource_monitor"]["sample_count"]) for row in rows
    )
    result["runtime_external_cpu_busy_percent_max"] = max(
        float(row["runtime_resource_monitor"]["external_cpu_busy_percent_max"])
        for row in rows
    )
    return result


def main() -> None:
    args = parse_args()
    policy_rows = {
        policy: load_point(args.root / "unique", policy, 2, "unique")
        for policy in POLICIES
    }
    reference_rows = policy_rows["sequential"]
    reference_signatures = {
        (
            tuple(row["sample_indices"]),
            row["retrieval_ids_sha256"],
            row["prediction_quality_sha256"],
        )
        for row in reference_rows
    }
    identical_outputs = all(
        (
            tuple(row["sample_indices"]),
            row["retrieval_ids_sha256"],
            row["prediction_quality_sha256"],
        )
        in reference_signatures
        for rows in policy_rows.values()
        for row in rows
    )
    aggregates = {policy: aggregate(rows) for policy, rows in policy_rows.items()}
    baseline = aggregates["sequential"]
    deltas = {
        policy: {
            key: 100.0
            * (aggregates[policy][f"{key}_mean"] / baseline[f"{key}_mean"] - 1.0)
            for key in PERFORMANCE_KEYS
        }
        for policy in POLICIES[1:]
    }
    reuse = load_point(args.root / "reuse", "prefill_start", 1, "reuse")[0]
    h2d_jobs = int(
        reuse["cpu_kv_transfer_latency"]["h2d_reload"]["worker_completed_jobs"]
    )
    summary = {
        "status": ("complete" if h2d_jobs > 0 and identical_outputs else "invalid"),
        "identical_samples_retrieval_and_predictions": identical_outputs,
        "policies": aggregates,
        "delta_percent_vs_sequential": deltas,
        "controlled_reuse": {
            "summary_path": reuse["summary_path"],
            "h2d_reload": reuse["cpu_kv_transfer_latency"]["h2d_reload"],
            "d2h_offload": reuse["cpu_kv_transfer_latency"]["d2h_offload"],
        },
    }
    output_path = args.root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
