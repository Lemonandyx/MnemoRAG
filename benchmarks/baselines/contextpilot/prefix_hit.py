#!/usr/bin/env python3
"""Measure ContextPilot prefix-cache hit rates on the full NQ test split.

Each invocation runs exactly one top-k point and creates a fresh vLLM engine.
The caller is responsible for launching separate processes for k=1,3,5,10 so
that prefix-cache state cannot leak between points.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
CONTEXTPILOT_ROOT = Path(
    os.environ.get("CONTEXTPILOT_ROOT", PROJECT_ROOT / "third_party/ContextPilot")
).resolve()
NQ_TEST_PATH = ASSET_ROOT / "qa/datasets/nq/test.jsonl"
DOC_IDS_PATH = Path(
    PROJECT_ROOT
    / "results/background_motivation/cache/nq_test_nprobe128_top10_doc_ids.npy"
)
CORPUS_PATH = ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
MODEL_PATH = ASSET_ROOT / "models/llama3-8B-instruct"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/contextpilot/prefix_hit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, required=True, choices=(1, 3, 5, 10))
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--stats-interval-s", type=float, default=1.0)
    parser.add_argument("--stats-window-requests", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--linkage-method", default="average")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_resources(args: argparse.Namespace) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    if len(available) < args.cpu_threads:
        raise RuntimeError(
            f"Requested {args.cpu_threads} CPU threads but only {len(available)} "
            "are available in this process affinity."
        )
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # This benchmark measures ContextPilot plus the native GPU prefix cache.
    # It intentionally does not enable MnemoRAG's CPU KV offload connector.
    for name in (
        "MNEMORAG_CPU_KV_BLOCKS",
        "MNEMORAG_CPU_KV_BLOCK_SIZE",
        "MNEMORAG_CPU_KV_EVICTION_POLICY",
    ):
        os.environ.pop(name, None)
    return selected


def load_nq_test() -> list[dict]:
    with NQ_TEST_PATH.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 3610:
        raise RuntimeError(
            f"Expected full NQ test split with 3610 rows, got {len(rows)}"
        )
    return rows


def load_doc_ids(topk: int, request_count: int) -> np.ndarray:
    values = np.load(DOC_IDS_PATH, mmap_mode="r")
    if values.shape != (request_count, 10):
        raise RuntimeError(
            f"Expected retrieval trace shape ({request_count}, 10), got {values.shape}"
        )
    return np.asarray(values[:, :topk], dtype=np.int64)


def load_or_extract_documents(
    all_top10_ids: np.ndarray, output_root: Path
) -> dict[int, str]:
    """Read only required rows from the 21M-document JSONL corpus once."""
    cache_path = output_root / "cache" / "nq_test_nprobe128_top10_documents.json"
    needed = {int(value) for value in all_top10_ids.reshape(-1)}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        docs = {int(key): value for key, value in raw.items()}
        missing = needed.difference(docs)
        if not missing:
            print(f"Loaded {len(docs):,} cached corpus documents", flush=True)
            return docs
        print(f"Document cache missing {len(missing):,} IDs; rebuilding", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    docs: dict[int, str] = {}
    maximum = max(needed)
    started = time.perf_counter()
    with CORPUS_PATH.open(encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            if line_index in needed:
                row = json.loads(line)
                row_id = int(row["id"])
                if row_id != line_index:
                    raise RuntimeError(
                        f"Corpus row mismatch at line {line_index}: id={row_id}"
                    )
                docs[line_index] = row["contents"]
            if line_index >= maximum and len(docs) == len(needed):
                break
            if line_index and line_index % 2_000_000 == 0:
                print(
                    f"Corpus scan {line_index:,} rows; found {len(docs):,}/"
                    f"{len(needed):,}",
                    flush=True,
                )
    missing = needed.difference(docs)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} retrieved document IDs")
    atomic_json(cache_path, {str(key): docs[key] for key in sorted(docs)})
    print(
        f"Extracted {len(docs):,} documents in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return docs


def build_contextpilot_plan(
    contexts: list[list[int]], args: argparse.Namespace
) -> tuple[list[list[int]], list[int], float]:
    sys.path.insert(0, str(CONTEXTPILOT_ROOT))
    from contextpilot import ContextPilot

    started = time.perf_counter()
    pilot = ContextPilot(
        use_gpu=False,
        alpha=args.alpha,
        linkage_method=args.linkage_method,
        batch_size=10_000,
    )
    # The official constructor inherits this field from ContextIndex but does
    # not expose it in ContextPilot.__init__. Setting the public field keeps
    # the official algorithm unchanged while honoring the 64-core limit.
    pilot.num_workers = args.cpu_threads
    reordered, order = pilot.reorder(contexts)
    elapsed = time.perf_counter() - started

    if sorted(order) != list(range(len(contexts))):
        raise RuntimeError("ContextPilot returned an invalid request permutation")
    for scheduled_index, original_index in enumerate(order):
        if sorted(reordered[scheduled_index]) != sorted(contexts[original_index]):
            raise RuntimeError(
                f"ContextPilot changed the retrieved document set for request "
                f"{original_index}"
            )
    return reordered, [int(value) for value in order], elapsed


def make_prompts(
    rows: list[dict],
    contexts: list[list[int]],
    reordered: list[list[int]],
    order: list[int],
    docs: dict[int, str],
    tokenizer,
    max_model_len: int,
) -> tuple[list[dict[str, list[int]]], list[int]]:
    sys.path.insert(0, str(CONTEXTPILOT_ROOT))
    from contextpilot.api import _build_system_prompt

    prompts: list[dict[str, list[int]]] = []
    token_lengths: list[int] = []
    for scheduled_index, original_index in enumerate(order):
        original_docs = [docs[doc_id] for doc_id in contexts[original_index]]
        reordered_docs = [docs[doc_id] for doc_id in reordered[scheduled_index]]
        messages = [
            {
                "role": "system",
                "content": _build_system_prompt(reordered_docs, original_docs),
            },
            {"role": "user", "content": rows[original_index]["question"]},
        ]
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        length = len(prompt_token_ids)
        if length + 1 > max_model_len:
            raise RuntimeError(
                f"Prompt for NQ row {original_index} has {length} tokens, exceeding "
                f"max_model_len={max_model_len}; refusing silent truncation"
            )
        # Passing token IDs avoids any ambiguity about special-token insertion
        # between Transformers and vLLM and makes the cache denominator exact.
        prompts.append({"prompt_token_ids": prompt_token_ids})
        token_lengths.append(length)
    return prompts, token_lengths


async def run_vllm(
    prompts: list[dict[str, list[int]]],
    order: list[int],
    token_lengths: list[int],
    args: argparse.Namespace,
    point_dir: Path,
) -> dict:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=True,
        disable_log_stats=False,
        seed=0,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    records: list[dict] = []
    system_windows: list[dict] = []
    recent: deque[tuple[int, int, int]] = deque()
    recent_prompt = 0
    recent_cached = 0
    next_stats_at = args.stats_interval_s
    started = time.perf_counter()
    try:
        for scheduled_index, prompt in enumerate(prompts):
            final_output = None
            # The server's vLLM branch records request position by converting
            # request_id to int, so IDs must remain decimal strings.
            request_id = str(scheduled_index)
            request_started = time.perf_counter()
            async for candidate in engine.generate(prompt, sampling, request_id):
                final_output = candidate
            if final_output is None:
                raise RuntimeError(f"vLLM produced no output for {request_id}")
            prompt_tokens = len(final_output.prompt_token_ids or ())
            cached_tokens = int(final_output.num_cached_tokens or 0)
            if prompt_tokens != token_lengths[scheduled_index]:
                raise RuntimeError(
                    f"Tokenizer/vLLM prompt length mismatch for {request_id}: "
                    f"{token_lengths[scheduled_index]} vs {prompt_tokens}"
                )
            hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0
            records.append(
                {
                    "scheduled_index": scheduled_index,
                    "original_index": order[scheduled_index],
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "hit_rate": hit_rate,
                    "latency_s": time.perf_counter() - request_started,
                }
            )
            recent.append((scheduled_index, prompt_tokens, cached_tokens))
            recent_prompt += prompt_tokens
            recent_cached += cached_tokens
            while len(recent) > args.stats_window_requests:
                _, old_prompt, old_cached = recent.popleft()
                recent_prompt -= old_prompt
                recent_cached -= old_cached

            elapsed = time.perf_counter() - started
            if elapsed >= next_stats_at:
                # Match vLLM CachingMetrics: one token-weighted system hit
                # rate over the most recent N requests. This is deliberately
                # not a statistic over individual request hit-rate values.
                system_windows.append(
                    {
                        "sample_index": len(system_windows),
                        "elapsed_s": elapsed,
                        "first_scheduled_index": recent[0][0],
                        "last_scheduled_index": recent[-1][0],
                        "request_count": len(recent),
                        "prompt_tokens": recent_prompt,
                        "cached_tokens": recent_cached,
                        "hit_rate": (
                            recent_cached / recent_prompt if recent_prompt else 0.0
                        ),
                    }
                )
                # AsyncLLM records scheduler statistics continuously but does
                # not periodically print them itself. Emit the native vLLM
                # line at the same point as an independent audit trail.
                await engine.do_log_stats()
                while next_stats_at <= elapsed:
                    next_stats_at += args.stats_interval_s
            if (scheduled_index + 1) % 100 == 0 or scheduled_index + 1 == len(prompts):
                partial_mean = statistics.fmean(r["hit_rate"] for r in records)
                print(
                    f"topk={args.topk} completed {scheduled_index + 1:,}/"
                    f"{len(prompts):,}; mean_hit={partial_mean:.6f}",
                    flush=True,
                )
    finally:
        engine.shutdown()

    wall_time = time.perf_counter() - started
    temporary = point_dir / "requests.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(point_dir / "requests.csv")

    if not system_windows:
        raise RuntimeError("No system-level prefix-cache samples were recorded")
    temporary = point_dir / "system_hit_rate_windows.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(system_windows[0]))
        writer.writeheader()
        writer.writerows(system_windows)
    temporary.replace(point_dir / "system_hit_rate_windows.csv")

    rates = [record["hit_rate"] for record in records]
    system_rates = [window["hit_rate"] for window in system_windows]
    total_prompt = sum(record["prompt_tokens"] for record in records)
    total_cached = sum(record["cached_tokens"] for record in records)
    return {
        "request_count": len(records),
        "average_hit_rate": statistics.fmean(system_rates),
        "max_hit_rate": max(system_rates),
        "system_hit_rate_sample_count": len(system_windows),
        "system_hit_rate_window_requests": args.stats_window_requests,
        "system_hit_rate_interval_s": args.stats_interval_s,
        "per_request_hit_rate_mean_diagnostic": statistics.fmean(rates),
        "per_request_hit_rate_max_diagnostic": max(rates),
        "requests_with_any_hit": sum(rate > 0 for rate in rates),
        "aggregate_cached_tokens": total_cached,
        "aggregate_prompt_tokens": total_prompt,
        "token_weighted_hit_rate": total_cached / total_prompt,
        "generation_wall_time_s": wall_time,
    }


async def async_main(args: argparse.Namespace, selected_cpus: list[int]) -> None:
    rows = load_nq_test()
    all_top10_ids = np.load(DOC_IDS_PATH, mmap_mode="r")
    doc_ids = load_doc_ids(args.topk, len(rows))
    contexts = [[int(value) for value in row] for row in doc_ids]
    point_dir = args.output_root / f"topk{args.topk}"
    point_dir.mkdir(parents=True, exist_ok=True)

    reordered, order, planning_time = build_contextpilot_plan(contexts, args)
    plan = {
        "topk": args.topk,
        "request_count": len(rows),
        "order": order,
        "reordered_doc_ids": reordered,
        "planning_time_s": planning_time,
    }
    atomic_json(point_dir / "contextpilot_plan.json", plan)
    if args.plan_only:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in plan.items()
                    if key not in {"order", "reordered_doc_ids"}
                },
                indent=2,
            )
        )
        return

    docs = load_or_extract_documents(all_top10_ids, args.output_root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    prompts, token_lengths = make_prompts(
        rows,
        contexts,
        reordered,
        order,
        docs,
        tokenizer,
        args.max_model_len,
    )
    metrics = await run_vllm(prompts, order, token_lengths, args, point_dir)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "status": "complete",
        "system": "ContextPilot",
        "contextpilot_source": str(CONTEXTPILOT_ROOT),
        "contextpilot_version": "0.4.2 source tree (__version__ reports 0.4.1)",
        "dataset": "NQ test",
        "dataset_path": str(NQ_TEST_PATH),
        "retrieval_trace": str(DOC_IDS_PATH),
        "retrieval_nprobe": 128,
        "topk": args.topk,
        "request_count": len(rows),
        "model": str(MODEL_PATH),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "physical_gpu_id": args.gpu_id,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "max_num_seqs": 1,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "contextpilot_alpha": args.alpha,
        "contextpilot_linkage_method": args.linkage_method,
        "contextpilot_planning_time_s": planning_time,
        "prompt_token_min": min(token_lengths),
        "prompt_token_mean": statistics.fmean(token_lengths),
        "prompt_token_max": max(token_lengths),
        "metric_definition": {
            "per_request_hit_rate": "vLLM num_cached_tokens / prompt token count",
            "system_window_hit_rate": (
                "sum cached tokens / sum prompt tokens over vLLM's most recent "
                f"{args.stats_window_requests} requests, sampled every "
                f"{args.stats_interval_s:g}s"
            ),
            "average_hit_rate": "arithmetic mean of system-window hit rates",
            "max_hit_rate": "maximum system-window hit rate",
            "token_weighted_hit_rate": "sum cached tokens / sum prompt tokens",
        },
        "cache_isolation": "fresh process and fresh AsyncLLM engine for this top-k point",
        "execution": "strict ContextPilot scheduled order; one request in flight",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "vllm": __import__("vllm").__version__,
        },
        "benchmark_script_sha256": source_sha,
        **metrics,
    }
    atomic_json(point_dir / "summary.json", summary)
    print("POINT_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    selected_cpus = configure_resources(args)
    asyncio.run(async_main(args, selected_cpus))


if __name__ == "__main__":
    main()
