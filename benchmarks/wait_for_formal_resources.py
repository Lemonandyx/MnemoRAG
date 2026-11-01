#!/usr/bin/env python3
"""Wait for the exact GPU1/socket0 resources required by formal experiments."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from .benchmark_point import resource_preflight
    from .run_full_e2e_campaign import atomic_write_json, verify_driver_topology
    from .runtime_resource_monitor import _gpu_processes
except ImportError:
    from benchmark_point import resource_preflight
    from run_full_e2e_campaign import atomic_write_json, verify_driver_topology
    from runtime_resource_monitor import _gpu_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--max-cpu-busy-percent", type=float, default=5.0)
    parser.add_argument("--max-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-gpu-util-percent", type=float, default=5.0)
    parser.add_argument("--consecutive-clean-samples", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one_sample(args: argparse.Namespace) -> dict:
    topology = verify_driver_topology(args.gpu_id)
    preflight_args = SimpleNamespace(
        gpu_id=str(args.gpu_id),
        require_clean_resources=False,
        max_preflight_cpu_busy_percent=float(args.max_cpu_busy_percent),
        max_preflight_cpu_max_busy_percent=float(args.max_cpu_busy_percent),
        max_preflight_gpu_memory_mib=float(args.max_gpu_memory_mib),
        max_preflight_gpu_util_percent=float(args.max_gpu_util_percent),
        max_preflight_load_per_cpu=0.10,
    )
    preflight = resource_preflight(preflight_args, list(topology["logical_cpu_ids"]))
    gpu_processes = _gpu_processes(str(args.gpu_id), os.getpid())
    violations = list(preflight["violations"])
    if gpu_processes:
        violations.append(f"GPU compute processes are present: {gpu_processes}")
    return {
        "sampled_at_unix_s": time.time(),
        "topology": topology,
        "preflight": preflight,
        "gpu_compute_processes": gpu_processes,
        "violations": violations,
        "clean": not violations,
    }


def main() -> None:
    args = parse_args()
    if args.consecutive_clean_samples < 1:
        raise ValueError("consecutive-clean-samples must be positive")
    history = []
    consecutive = 0
    while consecutive < args.consecutive_clean_samples:
        sample = one_sample(args)
        history.append(sample)
        history = history[-100:]
        consecutive = consecutive + 1 if sample["clean"] else 0
        atomic_write_json(
            args.output,
            {
                "status": (
                    "clean"
                    if consecutive >= args.consecutive_clean_samples
                    else "waiting"
                ),
                "required_consecutive_clean_samples": args.consecutive_clean_samples,
                "consecutive_clean_samples": consecutive,
                "samples": history,
            },
        )
        if consecutive < args.consecutive_clean_samples:
            time.sleep(args.retry_seconds)
    print(
        json.dumps(
            {
                "status": "clean",
                "consecutive_clean_samples": consecutive,
                "evidence": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
