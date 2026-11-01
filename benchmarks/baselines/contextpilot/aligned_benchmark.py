#!/usr/bin/env python3
"""Run ContextPilot on NQ-test under the shared MnemoRAG configuration.

This is intentionally an online benchmark: requests retain the source NQ-test
order, arrive according to a Poisson process, and are handed to one persistent
AsyncLLM engine.  ContextPilot sees each request only when it arrives and may
reorder documents *within* that request; it never sees future requests and
never globally permutes the request stream.

Each process runs one top-k point.  Launching a new process per point provides
a fresh vLLM engine and a cold prefix cache.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/contextpilot/aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, required=True, choices=(1, 3, 5, 10))
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--arrival-rate-rps", type=float, default=24.0)
    parser.add_argument("--arrival-seed", type=int, default=271828)
    parser.add_argument("--stats-interval-s", type=float, default=1.0)
    parser.add_argument("--stats-window-requests", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--linkage-method", default="average")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configure_resources(args: argparse.Namespace) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    if len(available) < args.cpu_threads:
        raise RuntimeError(
            f"Requested {args.cpu_threads} CPUs but affinity exposes "
            f"only {len(available)}"
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

    # ContextPilot is evaluated with its own cache policy.  Do not silently
    # inherit MnemoRAG's heat-aware policy or its CPU KV offload connector.
    os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = "lru"
    for name in (
        "MNEMORAG_CPU_KV_BLOCKS",
        "MNEMORAG_CPU_KV_BLOCK_SIZE",
        "MNEMORAG_CPU_KV_EVICTION_POLICY",
        "VLLM_KV_TRANSFER_CONFIG",
    ):
        os.environ.pop(name, None)
    return selected


def load_nq_test(max_requests: int | None) -> list[dict[str, Any]]:
    with NQ_TEST_PATH.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 3610:
        raise RuntimeError(f"Expected 3,610 NQ-test rows, got {len(rows)}")
    if max_requests is not None:
        if max_requests <= 0 or max_requests > len(rows):
            raise ValueError("--max-requests must be in [1, 3610]")
        rows = rows[:max_requests]
    return rows


def load_doc_ids(topk: int, request_count: int) -> np.ndarray:
    values = np.load(DOC_IDS_PATH, mmap_mode="r")
    if values.shape != (3610, 10):
        raise RuntimeError(f"Unexpected retrieval trace shape: {values.shape}")
    return np.asarray(values[:request_count, :topk], dtype=np.int64)


def load_or_extract_documents(request_count: int, output_root: Path) -> dict[int, str]:
    cache_path = output_root / "cache" / "nq_test_nprobe128_top10_documents.json"
    all_ids = np.load(DOC_IDS_PATH, mmap_mode="r")[:request_count]
    needed = {int(value) for value in np.asarray(all_ids).reshape(-1)}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        docs = {int(key): value for key, value in raw.items()}
        if not needed.difference(docs):
            return docs

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    docs: dict[int, str] = {}
    maximum = max(needed)
    with CORPUS_PATH.open(encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            if line_index in needed:
                row = json.loads(line)
                if int(row["id"]) != line_index:
                    raise RuntimeError(
                        f"Corpus row mismatch at {line_index}: id={row['id']}"
                    )
                docs[line_index] = row["contents"]
            if line_index >= maximum and len(docs) == len(needed):
                break
            if line_index and line_index % 2_000_000 == 0:
                print(
                    f"Corpus scan {line_index:,}; found {len(docs):,}/{len(needed):,}",
                    flush=True,
                )
    missing = needed.difference(docs)
    if missing:
        raise RuntimeError(f"Corpus cache is missing {len(missing)} documents")
    atomic_json(cache_path, {str(key): docs[key] for key in sorted(docs)})
    return docs


def poisson_offsets(count: int, rate: float, seed: int) -> np.ndarray:
    if rate <= 0:
        raise ValueError("arrival rate must be positive")
    offsets = np.zeros(count, dtype=np.float64)
    if count > 1:
        rng = np.random.default_rng(seed)
        offsets[1:] = np.cumsum(rng.exponential(1.0 / rate, size=count - 1))
    return offsets


class OnlineContextPilot:
    def __init__(self, args: argparse.Namespace):
        sys.path.insert(0, str(CONTEXTPILOT_ROOT))
        import contextpilot.server.live_index as live_index
        from contextpilot import ContextPilot
        from contextpilot.context_index.compute_distance_cpu import (
            compute_distance_single,
        )

        def safe_compute_distances_batch(
            queries: list[list[int]],
            targets: list[list[int]],
            alpha: float = 0.001,
            num_workers: int | None = None,
        ) -> np.ndarray:
            """Exact fallback for ContextPilot's unpickleable local worker.

            ContextPilot 0.4.2 defines its multiprocessing worker inside
            compute_distances_batch.  Once queries*targets reaches 1,000,
            multiprocessing must pickle that local function and aborts.  This
            fallback evaluates the identical compute_distance_single routine
            directly; it changes neither the distance nor the tree algorithm.
            """
            del num_workers
            distances = np.ones((len(queries), len(targets)), dtype=np.float32)
            for query_index, query in enumerate(queries):
                for target_index, target in enumerate(targets):
                    distances[query_index, target_index] = compute_distance_single(
                        query, target, alpha
                    )
            return distances

        # live_index imported the function by name, so patch that module-local
        # binding rather than editing the upstream ContextPilot checkout.
        live_index.compute_distances_batch = safe_compute_distances_batch

        self.pilot = ContextPilot(
            use_gpu=False,
            alpha=args.alpha,
            linkage_method=args.linkage_method,
            batch_size=10_000,
        )
        self.pilot.num_workers = args.cpu_threads

    def reorder(self, context: list[int]) -> list[int]:
        # The reference implementation prints a full incremental-build trace
        # for every call.  Suppress that diagnostic output without changing
        # the algorithm or timing the operation outside the request path.
        with contextlib.redirect_stdout(io.StringIO()):
            reordered_batch, original_indices = self.pilot.reorder(context)
        if original_indices != [0] or len(reordered_batch) != 1:
            raise RuntimeError(
                "Incremental ContextPilot returned an invalid single-request plan"
            )
        reordered = [int(value) for value in reordered_batch[0]]
        if sorted(reordered) != sorted(context):
            raise RuntimeError("ContextPilot changed the retrieved document set")
        return reordered


def format_reference(reordered_ids: list[int], docs: dict[int, str]) -> str:
    parts: list[str] = []
    for doc_number, doc_id in enumerate(reordered_ids, 1):
        contents = docs[doc_id]
        lines = contents.split("\n")
        title = lines[0]
        text = "\n".join(lines[1:])
        parts.append(f"Doc {doc_number}(Title: {title}) {text}\n")
    return "".join(parts)


def build_prompt(
    question: str,
    original_ids: list[int],
    reordered_ids: list[int],
    docs: dict[int, str],
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[str, int, bool]:
    reference = format_reference(reordered_ids, docs)
    system = (
        "Answer the question based on the given document. Only give me the "
        "answer and do not output any other words.\nDocuments:\n"
        f"{reference}"
    )
    position = {doc_id: index for index, doc_id in enumerate(reordered_ids, 1)}
    priority = [f"Doc {position[doc_id]}" for doc_id in original_ids]
    natural = [f"Doc {index}" for index in range(1, len(original_ids) + 1)]
    if priority != natural:
        system += (
            "\nNOTE: Please read the context in the following priority order: "
            f"{' → '.join(priority)} to answer the question."
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    truncated = len(token_ids) > max_prompt_tokens
    if truncated:
        # Same symmetric head/tail policy used by the shared benchmark prompt
        # builder, while reserving the configured generation budget.
        left = max_prompt_tokens // 2
        right = max_prompt_tokens - left
        prompt = tokenizer.decode(token_ids[:left] + token_ids[-right:])
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[:max_prompt_tokens]
            prompt = tokenizer.decode(token_ids)
    return prompt, len(token_ids), truncated


def get_logging_stat_logger(engine):
    manager = engine.logger_manager
    for aggregate_logger in manager.stat_loggers:
        if hasattr(aggregate_logger, "prefix_caching_metrics"):
            return aggregate_logger
        per_engine = getattr(aggregate_logger, "per_engine_stat_loggers", None)
        if per_engine:
            for logger in per_engine.values():
                if hasattr(logger, "prefix_caching_metrics"):
                    return logger
    raise RuntimeError("Could not locate vLLM LoggingStatLogger")


async def run_point(
    rows: list[dict[str, Any]],
    doc_ids: np.ndarray,
    docs: dict[int, str],
    args: argparse.Namespace,
    point_dir: Path,
) -> tuple[dict[str, Any], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        seed=0,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    stat_logger = get_logging_stat_logger(engine)
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    planner = OnlineContextPilot(args)
    arrivals = poisson_offsets(len(rows), args.arrival_rate_rps, args.arrival_seed)
    records: list[dict[str, Any] | None] = [None] * len(rows)
    system_windows: list[dict[str, Any]] = []
    reordered_doc_ids: list[list[int] | None] = [None] * len(rows)
    prompt_lengths: list[int] = []
    truncated_count = 0
    finished = asyncio.Event()
    started = time.perf_counter()

    async def collect(
        request_index: int,
        prompt: str,
        expected_prompt_tokens: int,
        target_arrival_s: float,
        actual_arrival_s: float,
        planning_s: float,
        was_truncated: bool,
    ) -> None:
        request_started = time.perf_counter()
        final_output = None
        async for candidate in engine.generate(prompt, sampling, str(request_index)):
            final_output = candidate
        if final_output is None:
            raise RuntimeError(f"vLLM produced no output for request {request_index}")
        prompt_tokens = len(final_output.prompt_token_ids or ())
        cached_tokens = int(final_output.num_cached_tokens or 0)
        records[request_index] = {
            "request_index": request_index,
            "target_arrival_s": target_arrival_s,
            "actual_arrival_s": actual_arrival_s,
            "arrival_lateness_s": actual_arrival_s - target_arrival_s,
            "contextpilot_planning_s": planning_s,
            "prompt_tokens_expected": expected_prompt_tokens,
            "prompt_tokens_vllm": prompt_tokens,
            "cached_tokens": cached_tokens,
            "request_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
            "prompt_truncated": int(was_truncated),
            "e2e_from_target_arrival_s": time.perf_counter()
            - started
            - target_arrival_s,
            "engine_response_s": time.perf_counter() - request_started,
        }

    async def sample_native_windows() -> None:
        last_lifetime_queries = -1
        while not finished.is_set():
            try:
                await asyncio.wait_for(finished.wait(), timeout=args.stats_interval_s)
            except asyncio.TimeoutError:
                pass
            await engine.do_log_stats()
            metrics = stat_logger.prefix_caching_metrics
            lifetime_queries = int(stat_logger.lifetime_prefix_cache_queries)
            if lifetime_queries == last_lifetime_queries:
                continue
            last_lifetime_queries = lifetime_queries
            if metrics.aggregated_requests < args.stats_window_requests:
                continue
            system_windows.append(
                {
                    "sample_index": len(system_windows),
                    "elapsed_s": time.perf_counter() - started,
                    "window_requests": int(metrics.aggregated_requests),
                    "window_prompt_tokens": int(metrics.aggregated_query_total),
                    "window_cached_tokens": int(metrics.aggregated_query_hit),
                    "hit_rate": float(metrics.hit_rate),
                    "lifetime_prompt_tokens": lifetime_queries,
                    "lifetime_cached_tokens": int(
                        stat_logger.lifetime_prefix_cache_hits
                    ),
                }
            )

    sampler_task = asyncio.create_task(sample_native_windows())
    tasks: list[asyncio.Task[None]] = []
    try:
        for request_index, (row, target_arrival_s) in enumerate(zip(rows, arrivals)):
            delay = started + float(target_arrival_s) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            original = [int(value) for value in doc_ids[request_index]]
            planning_started = time.perf_counter()
            reordered = await asyncio.to_thread(planner.reorder, original)
            planning_s = time.perf_counter() - planning_started
            reordered_doc_ids[request_index] = reordered
            prompt, prompt_tokens, was_truncated = build_prompt(
                row["question"],
                original,
                reordered,
                docs,
                tokenizer,
                args.max_model_len - args.max_tokens - 8,
            )
            prompt_lengths.append(prompt_tokens)
            truncated_count += int(was_truncated)
            actual_arrival_s = time.perf_counter() - started
            tasks.append(
                asyncio.create_task(
                    collect(
                        request_index,
                        prompt,
                        prompt_tokens,
                        float(target_arrival_s),
                        actual_arrival_s,
                        planning_s,
                        was_truncated,
                    )
                )
            )
            if (request_index + 1) % 250 == 0:
                complete = sum(record is not None for record in records)
                print(
                    f"topk={args.topk}: submitted {request_index + 1:,}/"
                    f"{len(rows):,}, completed {complete:,}",
                    flush=True,
                )
        await asyncio.gather(*tasks)
    finally:
        finished.set()
        await sampler_task
        engine.shutdown()

    complete_records = [record for record in records if record is not None]
    if len(complete_records) != len(rows):
        raise RuntimeError(
            f"Only {len(complete_records)}/{len(rows)} requests completed"
        )
    if len(rows) >= args.stats_window_requests and not system_windows:
        raise RuntimeError("No full native 1,000-request cache window was recorded")

    total_prompt = sum(record["prompt_tokens_vllm"] for record in complete_records)
    total_cached = sum(record["cached_tokens"] for record in complete_records)
    full_rates = [window["hit_rate"] for window in system_windows]
    per_request_rates = [record["request_hit_rate"] for record in complete_records]
    reordered_count = sum(
        [int(value) for value in doc_ids[index]] != reordered_doc_ids[index]
        for index in range(len(rows))
    )
    metrics = {
        "request_count": len(rows),
        "arrival_trace_duration_s": float(arrivals[-1]) if len(arrivals) else 0.0,
        "wall_time_s": time.perf_counter() - started,
        "average_hit_rate": statistics.fmean(full_rates) if full_rates else None,
        "max_hit_rate": max(full_rates) if full_rates else None,
        "full_system_window_sample_count": len(full_rates),
        "native_window_request_target": args.stats_window_requests,
        "token_weighted_lifetime_hit_rate": (
            total_cached / total_prompt if total_prompt else 0.0
        ),
        "aggregate_cached_tokens": total_cached,
        "aggregate_prompt_tokens": total_prompt,
        "per_request_hit_rate_mean_diagnostic": statistics.fmean(per_request_rates),
        "per_request_hit_rate_max_diagnostic": max(per_request_rates),
        "requests_with_any_hit": sum(rate > 0 for rate in per_request_rates),
        "contextpilot_reordered_request_count": reordered_count,
        "contextpilot_reordered_request_rate": reordered_count / len(rows),
        "prompt_token_min": min(prompt_lengths),
        "prompt_token_mean": statistics.fmean(prompt_lengths),
        "prompt_token_max": max(prompt_lengths),
        "prompt_truncated_count": truncated_count,
        "arrival_lateness_mean_s": statistics.fmean(
            record["arrival_lateness_s"] for record in complete_records
        ),
        "arrival_lateness_max_s": max(
            record["arrival_lateness_s"] for record in complete_records
        ),
        "contextpilot_planning_mean_s": statistics.fmean(
            record["contextpilot_planning_s"] for record in complete_records
        ),
        "contextpilot_planning_p99_s": float(
            np.percentile(
                [record["contextpilot_planning_s"] for record in complete_records],
                99,
            )
        ),
    }
    return (
        metrics,
        [int(round(value * 1_000_000)) for value in arrivals],
        complete_records,
        system_windows,
    )


async def async_main(args: argparse.Namespace, selected_cpus: list[int]) -> None:
    rows = load_nq_test(args.max_requests)
    doc_ids = load_doc_ids(args.topk, len(rows))
    docs = load_or_extract_documents(len(rows), args.output_root)
    point_dir = args.output_root / f"topk{args.topk}"
    point_dir.mkdir(parents=True, exist_ok=True)

    metrics, arrival_us, records, system_windows = await run_point(
        rows, doc_ids, docs, args, point_dir
    )
    write_csv(point_dir / "requests.csv", records)
    write_csv(point_dir / "system_hit_rate_windows.csv", system_windows)
    reordered_ids = [
        [int(value) for value in doc_ids[record["request_index"]]] for record in records
    ]
    # The detailed request CSV contains all measurements; this compact trace
    # makes the exact Poisson workload independently replayable.
    atomic_json(
        point_dir / "arrival_trace.json",
        {
            "unit": "microseconds from benchmark start",
            "arrival_offsets": arrival_us,
            "source_doc_ids": reordered_ids,
        },
    )
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "status": "complete",
        "system": "ContextPilot",
        "dataset": "NQ test",
        "dataset_path": str(NQ_TEST_PATH),
        "retrieval_trace": str(DOC_IDS_PATH),
        "retrieval_nprobe": 128,
        "topk": args.topk,
        "model": str(MODEL_PATH),
        "vllm_version": __import__("vllm").__version__,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "physical_gpu_id": args.gpu_id,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "arrival_process": "Poisson; exponential inter-arrivals; not normalized",
        "arrival_rate_rps": args.arrival_rate_rps,
        "arrival_seed": args.arrival_seed,
        "contextpilot_mode": (
            "official incremental online reorder; source request order retained; "
            "no future visibility and no global request permutation"
        ),
        "contextpilot_alpha": args.alpha,
        "contextpilot_linkage_method": args.linkage_method,
        "contextpilot_cpu_distance_workaround": (
            "exact sequential evaluation of compute_distance_single; required "
            "because ContextPilot 0.4.2's local multiprocessing worker cannot "
            "be pickled once query-target pairs reach 1000"
        ),
        "prompt_contract": (
            "MnemoRAG one-pass system/user prompt and document format; priority "
            "hint only when ContextPilot changes document order"
        ),
        "kv_policy": "native vLLM LRU; no MnemoRAG heat policy; no CPU offload",
        "cache_isolation": "fresh process and cold AsyncLLM engine per top-k",
        "metric_definition": {
            "average_hit_rate": (
                "arithmetic mean of native vLLM recent-1000-request token-weighted "
                "hit rates, sampled once per second only after the window is full"
            ),
            "max_hit_rate": (
                "maximum of those full native recent-1000-request system windows"
            ),
            "token_weighted_lifetime_hit_rate": (
                "sum num_cached_tokens / sum prompt tokens across all requests"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
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
