#!/usr/bin/env python3
"""Validate one resumable prefill/offload campaign point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_dir", type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--trace-mode", choices=("unique", "reuse"), required=True)
    parser.add_argument("--cpu-set", required=True)
    return parser.parse_args()


def validate_point(point_dir: Path, policy: str, trace_mode: str, cpu_set: str) -> dict:
    summaries = sorted(point_dir.glob("**/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one summary under {point_dir}, found {len(summaries)}"
        )
    row = json.loads(summaries[0].read_text(encoding="utf-8"))
    expected_cpus = sorted(int(value) for value in cpu_set.split(",") if value)
    topology = row.get("cpu_topology") or {}
    preflight = row.get("resource_preflight") or {}
    runtime = row.get("runtime_resource_monitor") or {}
    transfer = row.get("cpu_kv_transfer_latency") or {}
    d2h = transfer.get("d2h_offload") or {}
    h2d = transfer.get("h2d_reload") or {}
    expected = {
        "request_cycle_size": 0 if trace_mode == "unique" else 80,
        "components": "full" if trace_mode == "unique" else "async_baseline",
        "max_tokens": 32 if trace_mode == "unique" else 8,
        "query_base_exact_lookup": trace_mode == "unique",
    }
    failures = []
    checks = {
        "status": row.get("status") == "complete",
        "policy": row.get("async_retrieval_policy") == policy,
        "topk": int(row.get("topk", -1)) == 5,
        "nprobe": int(row.get("nprobe", -1)) == 128,
        "cpu_affinity": sorted(row.get("cpu_affinity") or []) == expected_cpus,
        "physical_cores": topology.get("physical_core_count") == 64,
        "no_smt": topology.get("smt_siblings_selected") is False,
        "preflight": not preflight.get("violations"),
        "runtime_clean": runtime.get("status") == "clean",
        "runtime_continuous": runtime.get("continuous") is True,
        "runtime_samples": int(runtime.get("sample_count", 0)) >= 2,
        "runtime_gpu": int(runtime.get("foreign_gpu_process_count_max", -1)) == 0,
        "runtime_cpu": (
            runtime.get("external_cpu_busy_percent_max") is not None
            and float(runtime["external_cpu_busy_percent_max"]) <= 5.0
        ),
        "cpu_kv_enabled": bool((row.get("cpu_kv_offload") or {}).get("enabled")),
        "d2h": int(d2h.get("worker_completed_jobs", 0)) > 0,
        "retrieval_hash": bool(row.get("retrieval_ids_sha256")),
        "prediction_hash": bool(row.get("prediction_quality_sha256")),
        "native_prefill_event": row.get("native_prefill_admission_event") is True,
    }
    for key, value in expected.items():
        checks[key] = row.get(key) == value
    if trace_mode == "reuse":
        checks["h2d"] = int(h2d.get("worker_completed_jobs", 0)) > 0
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(
            f"invalid point {summaries[0]}: failed {', '.join(failures)}"
        )
    return {
        "status": "valid",
        "summary_path": str(summaries[0]),
        "policy": policy,
        "trace_mode": trace_mode,
        "runtime_samples": runtime["sample_count"],
        "runtime_external_cpu_busy_percent_max": runtime[
            "external_cpu_busy_percent_max"
        ],
    }


def main() -> None:
    args = parse_args()
    try:
        result = validate_point(
            args.point_dir, args.policy, args.trace_mode, args.cpu_set
        )
    except RuntimeError as error:
        print(f"POINT_INVALID {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
