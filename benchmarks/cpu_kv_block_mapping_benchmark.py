#!/usr/bin/env python3
"""Benchmark CPU preparation of GPU/CPU KV block mappings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids


def reference_expand(block_ids, block_size_factor, output, skip_count=0):
    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(block_size_factor)
    output_idx = 0
    for index, block_id in enumerate(block_ids):
        offsets = first_range if index == 0 else full_range
        output_end = output_idx + len(offsets)
        output[output_idx:output_end] = block_id * block_size_factor + offsets
        output_idx = output_end


def measure(function, block_ids, factor, output, repetitions):
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function(block_ids, factor, output)
        samples.append(time.perf_counter() - started)
    return {
        "mean_ms": 1000.0 * statistics.fmean(samples),
        "p95_ms": 1000.0 * float(np.percentile(samples, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for block_count in args.blocks:
        for factor in args.factors:
            block_ids = np.arange(block_count, dtype=np.int64) * 3
            reference = np.empty(block_count * factor, dtype=np.int64)
            optimized = np.empty_like(reference)
            reference_expand(block_ids, factor, reference)
            expand_block_ids(block_ids, factor, optimized)
            exact = bool(np.array_equal(reference, optimized))
            if not exact:
                raise RuntimeError("optimized block expansion changed the mapping")
            old = measure(
                reference_expand,
                block_ids,
                factor,
                reference,
                args.repetitions,
            )
            new = measure(
                expand_block_ids,
                block_ids,
                factor,
                optimized,
                args.repetitions,
            )
            rows.append(
                {
                    "blocks": block_count,
                    "block_size_factor": factor,
                    "exact": exact,
                    "reference": old,
                    "optimized": new,
                    "mean_speedup": old["mean_ms"] / new["mean_ms"],
                }
            )

    summary = {
        "status": "complete",
        "repetitions": args.repetitions,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
