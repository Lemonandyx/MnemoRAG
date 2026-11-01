#!/usr/bin/env python3
"""Validate and aggregate the top-k=1 H2D scheduling ABBA experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

MODES = ("async", "adaptive")
RUNS = (("async", 1), ("adaptive", 1), ("adaptive", 2), ("async", 2))
METRICS = (
    "wall_time_s",
    "throughput_qps",
    "request_latency_mean_s",
    "request_latency_p95_s",
    "retrieval_mean_s",
    "generation_e2e_mean_s",
    "request_prefill_service_mean_s",
    "batch_prefill_window_mean_s",
    "h2d_metadata_delivery_mean_s",
    "h2d_worker_submit_mean_s",
    "h2d_cuda_mean_s",
    "h2d_worker_observed_mean_s",
    "h2d_defer_delay_mean_s",
    "h2d_blocking_wait_mean_s",
    "h2d_scheduler_wait_mean_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def one_summary(root: Path, mode: str, replicate: int) -> tuple[Path, dict]:
    paths = sorted((root / f"{mode}_r{replicate}").glob("**/summary.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one summary for {mode}_r{replicate}, found {len(paths)}"
        )
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def validate(path: Path, data: dict, mode: str) -> None:
    h2d = nested(data, "cpu_kv_transfer_latency", "h2d_reload", default={})
    offload = data.get("cpu_kv_offload") or {}
    topology = data.get("cpu_topology") or {}
    expected = {
        "status": data.get("status") == "complete",
        "topk": int(data.get("topk", -1)) == 1,
        "nprobe": int(data.get("nprobe", -1)) == 128,
        "num_requests": int(data.get("num_requests", -1)) == 640,
        "request_cycle_size": int(data.get("request_cycle_size", -1)) == 320,
        "retrieval_batch_size": int(data.get("retrieval_batch_size", -1)) == 64,
        "microbatch_size": int(data.get("microbatch_size", -1)) == 64,
        "h2d_mode": offload.get("h2d_scheduling") == mode,
        "cpu_kv_enabled": bool(offload.get("enabled")),
        "cpu_kv_capacity": abs(float(offload.get("capacity_gb", 0.0)) - 256.0) < 0.01,
        "h2d_samples": int(h2d.get("worker_completed_jobs", 0)) > 0,
        "d2h_samples": int(
            nested(
                data,
                "cpu_kv_transfer_latency",
                "d2h_offload",
                "worker_completed_jobs",
                default=0,
            )
        )
        > 0,
        "physical_cores": int(topology.get("physical_core_count", -1)) == 64,
        "no_smt_duplicates": not bool(topology.get("smt_siblings_selected")),
    }
    failed = [name for name, valid in expected.items() if not valid]
    if failed:
        raise RuntimeError(f"invalid point {path}: {', '.join(failed)}")


def row(path: Path, data: dict, mode: str, replicate: int) -> dict[str, Any]:
    h2d = nested(data, "cpu_kv_transfer_latency", "h2d_reload", default={})
    request_stages = nested(data, "stage_latency", "request", "stages", default={})
    batch_stages = nested(
        data, "stage_latency", "batch", "scheduler_trace_stages", default={}
    )
    return {
        "mode": mode,
        "replicate": replicate,
        "summary_path": str(path),
        "wall_time_s": float(data["wall_time_s"]),
        "throughput_qps": float(data["throughput_qps"]),
        "request_latency_mean_s": float(data["request_latency_mean_s"]),
        "request_latency_p95_s": float(data["request_latency_p95_s"]),
        "retrieval_mean_s": float(nested(data, "stage_latency", "retrieval_mean_s")),
        "generation_e2e_mean_s": float(
            nested(data, "stage_latency", "generation_e2e_mean_s")
        ),
        "request_prefill_service_mean_s": float(
            nested(request_stages, "prefill_service_s", "mean_s", default=0.0)
        ),
        "batch_prefill_window_mean_s": float(
            nested(batch_stages, "prefill", "mean_s", default=0.0)
        ),
        "prefix_hit_rate_mean": float(data.get("prefix_request_hit_rate_mean", 0.0)),
        "h2d_jobs": int(h2d.get("worker_completed_jobs", 0)),
        "h2d_requests": int(h2d.get("worker_completed_requests", 0)),
        "h2d_async_jobs": int(h2d.get("async_jobs", 0)),
        "h2d_deferred_sync_jobs": int(h2d.get("deferred_sync_jobs", 0)),
        "h2d_metadata_delivery_mean_s": float(h2d.get("metadata_delivery_mean_s", 0.0)),
        "h2d_worker_submit_mean_s": float(h2d.get("worker_submit_mean_s", 0.0)),
        "h2d_cuda_mean_s": float(h2d.get("cuda_mean_s", 0.0)),
        "h2d_worker_observed_mean_s": float(h2d.get("worker_observed_mean_s", 0.0)),
        "h2d_defer_delay_mean_s": float(h2d.get("defer_delay_mean_s", 0.0)),
        "h2d_blocking_wait_mean_s": float(h2d.get("blocking_wait_mean_s", 0.0)),
        "h2d_scheduler_wait_mean_s": float(h2d.get("scheduler_wait_mean_s", 0.0)),
        "retrieval_ids_sha256": data.get("retrieval_ids_sha256"),
        "prediction_quality_sha256": data.get("prediction_quality_sha256"),
        "f1": float(nested(data, "metrics", "f1", default=0.0)),
        "accuracy": float(nested(data, "metrics", "acc", default=0.0)),
        "recall": float(nested(data, "metrics", "retrieval_recall_top1", default=0.0)),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for mode, replicate in RUNS:
        path, data = one_summary(args.root, mode, replicate)
        validate(path, data, mode)
        rows.append(row(path, data, mode, replicate))

    retrieval_signatures = {item["retrieval_ids_sha256"] for item in rows}
    if len(retrieval_signatures) != 1:
        raise RuntimeError("retrieval outputs differ between ABBA points")
    prediction_signatures = {item["prediction_quality_sha256"] for item in rows}
    quality_spread = {
        key: max(float(item[key]) for item in rows)
        - min(float(item[key]) for item in rows)
        for key in ("f1", "accuracy", "recall")
    }
    if (
        quality_spread["f1"] > 0.002
        or quality_spread["accuracy"] > 0.002
        or quality_spread["recall"] > 0.0
    ):
        raise RuntimeError(f"quality drift exceeds tolerance: {quality_spread}")

    aggregates: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_rows = [item for item in rows if item["mode"] == mode]
        aggregates[mode] = {
            "replicates": len(mode_rows),
            **{
                key: statistics.fmean(float(item[key]) for item in mode_rows)
                for key in METRICS
            },
            "prefix_hit_rate_mean": statistics.fmean(
                float(item["prefix_hit_rate_mean"]) for item in mode_rows
            ),
            "h2d_jobs": sum(int(item["h2d_jobs"]) for item in mode_rows),
            "h2d_requests": sum(int(item["h2d_requests"]) for item in mode_rows),
            "h2d_async_jobs": sum(int(item["h2d_async_jobs"]) for item in mode_rows),
            "h2d_deferred_sync_jobs": sum(
                int(item["h2d_deferred_sync_jobs"]) for item in mode_rows
            ),
        }

    baseline = aggregates["async"]
    candidate = aggregates["adaptive"]
    delta_percent = {
        key: 100.0 * (float(candidate[key]) / float(baseline[key]) - 1.0)
        for key in METRICS
        if float(baseline[key]) != 0.0
    }
    summary = {
        "status": "complete",
        "design": "ABBA",
        "identical_retrieval": True,
        "identical_predictions": len(prediction_signatures) == 1,
        "quality_spread": quality_spread,
        "runs": rows,
        "aggregates": aggregates,
        "adaptive_delta_percent_vs_async": delta_percent,
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
