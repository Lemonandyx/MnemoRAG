#!/usr/bin/env python3
"""Measure bounded dense document-hotness storage against Python Counter."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from faiss_numa_benchmark import cpu_topology, resource_preflight  # noqa: E402

from mnemorag.pipeline.utils import DocumentPool  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("counter", "dense"), required=True)
    parser.add_argument("--num-documents", type=int, default=21_015_324)
    parser.add_argument("--unique-updates", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--gpu-id", default="2")
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-cpu-mean-busy", type=float, default=20.0)
    parser.add_argument("--max-cpu-max-busy", type=float, default=50.0)
    parser.add_argument("--max-load-per-cpu", type=float, default=0.10)
    parser.add_argument("--max-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def process_memory_kib():
    values = {}
    with open("/proc/self/status", encoding="utf-8") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM", "VmSize"}:
                values[name] = int(value.split()[0])
    return values


def main():
    args = parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != args.cpu_threads:
        raise RuntimeError(
            f"Affinity exposes {len(affinity)} CPUs; expected {args.cpu_threads}"
        )
    topology = cpu_topology(affinity)
    if topology["physical_core_count"] != args.cpu_threads:
        raise RuntimeError("Selected CPUs are not distinct physical cores")
    if args.unique_updates > args.num_documents:
        raise ValueError("unique_updates must not exceed num_documents")
    preflight = resource_preflight(args, affinity)

    pool = DocumentPool(
        enable_hotness=True,
        num_documents=args.num_documents if args.mode == "dense" else None,
    )
    gc.collect()
    before = process_memory_kib()

    multiplier = 7_919
    started = time.perf_counter()
    for start in range(0, args.unique_updates, args.batch_size):
        stop = min(start + args.batch_size, args.unique_updates)
        doc_ids = [
            (index * multiplier) % args.num_documents for index in range(start, stop)
        ]
        pool.update(doc_ids)
    elapsed = time.perf_counter() - started
    gc.collect()
    after = process_memory_kib()

    sample_ids = [
        (index * multiplier) % args.num_documents
        for index in (0, args.unique_updates // 2, args.unique_updates - 1)
    ]
    if any(pool.get_count(doc_id) != 1 for doc_id in sample_ids):
        raise RuntimeError("Document counts changed during the benchmark")
    if pool.total_docs() != args.unique_updates:
        raise RuntimeError(
            f"Expected {args.unique_updates} unique IDs, got {pool.total_docs()}"
        )

    result = {
        "status": "complete",
        "mode": args.mode,
        "num_documents": args.num_documents,
        "unique_updates": args.unique_updates,
        "batch_size": args.batch_size,
        "elapsed_s": elapsed,
        "updates_per_s": args.unique_updates / elapsed,
        "dense_count_bytes": pool.dense_count_bytes,
        "fallback_entries": len(pool.doc_counter),
        "total_docs": pool.total_docs(),
        "memory_before_kib": before,
        "memory_after_kib": after,
        "rss_delta_bytes": (after["VmRSS"] - before["VmRSS"]) * 1024,
        "cpu_topology": topology,
        "resource_preflight": preflight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DOCUMENT_POOL_BENCHMARK_COMPLETE " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
