#!/usr/bin/env python3
"""Benchmark cached versus per-call KV policy environment lookups."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from benchmark_point import resource_preflight, validate_physical_cpu_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path(
    os.environ.get("MNEMORAG_VLLM_ROOT", PROJECT_ROOT / "third_party/vllm")
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=500_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-preflight-cpu-busy-percent", type=float, default=20.0)
    parser.add_argument(
        "--max-preflight-cpu-max-busy-percent", type=float, default=50.0
    )
    parser.add_argument("--max-preflight-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-preflight-gpu-util-percent", type=float, default=10.0)
    parser.add_argument("--max-preflight-load-per-cpu", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "results/evaluation/kv_priority/benchmark.json"),
    )
    return parser.parse_args()


def baseline_priority(queue, block) -> float:
    """Previous implementation, including environment reads per call."""
    if queue.eviction_policy == "fifo":
        return float(block.fifo_insert_seq)
    if queue.eviction_policy in {"heat", "chain_heat"} and (
        os.environ.get("VLLM_KV_PROFILE_NEXT_USE_PATH")
        or os.environ.get("VLLM_KV_ORACLE_NEXT_USE_PATH")
    ):
        if math.isinf(block.oracle_next_use_request):
            return -1.0e15
        return -block.oracle_next_use_request + min(block.heat, 1.0e6) * 1.0e-6
    if queue.eviction_policy in {"utility", "chain_utility", "chain_pgdsf"}:
        return block.utility_score
    if (
        queue.eviction_policy in {"heat", "chain_heat"}
        and os.environ.get("VLLM_KV_HEAT_FREQUENCY_LRU", "0") == "1"
    ):
        if block.heat <= 0:
            return 0.0
        bounded_recency = block.last_used_ts / (1.0 + block.last_used_ts)
        return block.heat + 0.5 * bounded_recency
    if (
        queue.eviction_policy in {"heat", "chain_heat"}
        and block.is_tail_probation
        and block.heat < 2
    ):
        return 0.0
    if (
        queue.eviction_policy == "heat"
        and os.environ.get("VLLM_KV_HEAT_PURE_LFU", "0") == "1"
    ):
        return block.heat
    if block.heat <= 0:
        return float("-inf")
    return math.log(block.heat) + queue.heat_decay_lambda * block.last_used_ts


def measure(function, queue, block, calls: int) -> tuple[float, float]:
    checksum = 0.0
    started = time.perf_counter()
    for _ in range(calls):
        checksum += function(queue, block)
    return time.perf_counter() - started, checksum


def main() -> None:
    args = parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != args.cpu_threads:
        raise RuntimeError(
            f"Affinity exposes {len(affinity)} CPUs; expected {args.cpu_threads}"
        )
    args.cpu_topology = validate_physical_cpu_ids(affinity)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = "heat"
    for variable in (
        "VLLM_KV_PROFILE_NEXT_USE_PATH",
        "VLLM_KV_ORACLE_NEXT_USE_PATH",
        "VLLM_KV_HEAT_FREQUENCY_LRU",
        "VLLM_KV_HEAT_PURE_LFU",
    ):
        os.environ.pop(variable, None)
    args.resource_preflight = resource_preflight(args, affinity)

    sys.path.insert(0, str(VLLM_ROOT))
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

    queue = FreeKVCacheBlockQueue([KVCacheBlock(0)])
    block = queue._all_blocks[0]
    block.heat = 7
    block.last_used_ts = 123.0

    methods = {
        "environment_per_call": baseline_priority,
        "cached_configuration": lambda q, b: q.calc_priority(b),
    }
    records = {name: [] for name in methods}
    checksums = {name: [] for name in methods}
    for replicate in range(args.replicates):
        order = list(methods)
        if replicate % 2:
            order.reverse()
        for name in order:
            elapsed, checksum = measure(methods[name], queue, block, args.calls)
            records[name].append(elapsed)
            checksums[name].append(checksum)

    expected = checksums["environment_per_call"][0]
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
        for values in checksums.values()
        for value in values
    ):
        raise RuntimeError("Priority checksum changed")

    modes = {}
    for name, values in records.items():
        modes[name] = {
            "elapsed_s": values,
            "mean_elapsed_s": statistics.fmean(values),
            "mean_ns_per_call": (statistics.fmean(values) * 1.0e9 / args.calls),
            "checksum": checksums[name][0],
        }
    baseline = modes["environment_per_call"]["mean_elapsed_s"]
    optimized = modes["cached_configuration"]["mean_elapsed_s"]
    result = {
        "status": "complete",
        "calls_per_replicate": args.calls,
        "replicates": args.replicates,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": affinity,
        "cpu_topology": args.cpu_topology,
        "physical_gpu_id": args.gpu_id,
        "resource_preflight": args.resource_preflight,
        "modes": modes,
        "speedup": baseline / optimized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print("KV_PRIORITY_BENCHMARK_COMPLETE " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
