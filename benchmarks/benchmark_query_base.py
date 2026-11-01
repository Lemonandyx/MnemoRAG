#!/usr/bin/env python3
"""Benchmark MnemoRAG Query Base nearest-query matching on a CUDA GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 256])
    parser.add_argument(
        "--dtypes",
        choices=("float16", "float32"),
        nargs="+",
        default=["float16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations-single", type=int, default=200)
    parser.add_argument("--iterations-batch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)),
        "std_ms": float(statistics.pstdev(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def load_embeddings(path: Path, capacity: int) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding array, got {source.shape}")
    indices = np.arange(capacity) % source.shape[0]
    return np.asarray(source[indices], dtype=np.float32)


@torch.no_grad()
def search_current(
    query: torch.Tensor, history_list: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce QueryPool.search_similar_batch from the current implementation."""
    history = torch.stack(history_list, dim=0)
    query_normalized = F.normalize(query, dim=1)
    history_normalized = F.normalize(history, dim=1)
    similarity = query_normalized @ history_normalized.T
    return torch.topk(similarity, k=1, dim=1)


@torch.no_grad()
def search_contiguous(
    query: torch.Tensor, history_normalized: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search a resident, contiguous, pre-normalized Query Base matrix."""
    query_normalized = F.normalize(query, dim=1)
    similarity = query_normalized @ history_normalized.T
    return torch.topk(similarity, k=1, dim=1)


def measure(
    search, query: torch.Tensor, iterations: int, warmup: int
) -> tuple[dict, dict]:
    for _ in range(warmup):
        scores, indices = search(query)
        scores.cpu()
        indices.cpu()
    torch.cuda.synchronize()

    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        start_event.record()
        scores, indices = search(query)
        end_event.record()
        scores.cpu()
        indices.cpu()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        gpu_ms.append(float(start_event.elapsed_time(end_event)))
    return summarize(wall_ms), summarize(gpu_ms)


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base = load_embeddings(args.embeddings, args.capacity)
    rng = np.random.default_rng(args.seed)
    results: list[dict] = []

    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        history = torch.from_numpy(base).to(device="cuda", dtype=dtype)
        history_list = list(history.unbind(dim=0))
        history_normalized = F.normalize(history, dim=1).contiguous()
        torch.cuda.synchronize()

        for batch_size in args.batch_sizes:
            query_indices = rng.integers(0, args.capacity, size=batch_size)
            query = torch.from_numpy(base[query_indices]).to(device="cuda", dtype=dtype)
            iterations = (
                args.iterations_single if batch_size == 1 else args.iterations_batch
            )

            for implementation, search in (
                (
                    "current_query_pool",
                    partial(search_current, history_list=history_list),
                ),
                (
                    "contiguous_prenormalized",
                    partial(
                        search_contiguous,
                        history_normalized=history_normalized,
                    ),
                ),
            ):
                wall, gpu = measure(search, query, iterations, args.warmup)
                results.append(
                    {
                        "implementation": implementation,
                        "dtype": dtype_name,
                        "capacity": args.capacity,
                        "dimension": int(base.shape[1]),
                        "batch_size": batch_size,
                        "iterations": iterations,
                        "wall": wall,
                        "gpu": gpu,
                        "per_query_mean_ms": wall["mean_ms"] / batch_size,
                        "queries_per_second": batch_size * 1000.0 / wall["mean_ms"],
                    }
                )

        del query, search, history_normalized, history_list, history
        torch.cuda.empty_cache()

    embedding_bytes = {
        dtype_name: args.capacity
        * base.shape[1]
        * torch.tensor([], dtype=getattr(torch, dtype_name)).element_size()
        for dtype_name in args.dtypes
    }
    doc_id_bytes = args.capacity * args.topk * np.dtype(np.int32).itemsize
    payload = {
        "experiment": "mnemorag_query_base_matching",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arguments": vars(args)
        | {"embeddings": str(args.embeddings), "output": str(args.output)},
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_properties": str(torch.cuda.get_device_properties(0)),
            "cpu_affinity": command_output(["taskset", "-cp", str(os.getpid())]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader",
                ]
            ),
        },
        "storage": {
            "embedding_bytes": embedding_bytes,
            "topk_int32_document_id_bytes": doc_id_bytes,
            "note": "Excludes Python object, text, score, cluster-id, and allocator overhead.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
