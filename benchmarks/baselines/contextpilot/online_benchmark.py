#!/usr/bin/env python3
"""Benchmark the official ContextPilot online path on the full NQ test split.

The ContextPilot algorithm is never imported or reimplemented here.  Each
request is sent to the official HTTP server's ``/reorder`` endpoint, and the
returned request id is forwarded to the official vLLM 0.15.x eviction hook.

Two workload modes are supported:

* ``sequential``: the next request arrives after the previous response.  This
  is the online A/B convention used by ``examples/mem0_bench_simple.py``.
* ``poisson``: requests retain NQ source order but arrive according to an
  exponential inter-arrival process.  This is the RPS extension used for the
  cross-system serving comparison; it is not presented as a paper parameter.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
OFFICIAL_ROOT = Path(
    os.environ.get("CONTEXTPILOT_ROOT", PROJECT_ROOT / "third_party/ContextPilot")
).resolve()
NQ_TEST_PATH = ASSET_ROOT / "qa/datasets/nq/test.jsonl"
DOC_IDS_PATH = Path(
    PROJECT_ROOT
    / "results/background_motivation/cache/nq_test_nprobe128_top10_doc_ids.npy"
)
DOCUMENT_CACHE = Path(
    PROJECT_ROOT
    / "results/contextpilot/aligned/cache/nq_test_nprobe128_top10_documents.json"
)
MODEL_PATH = ASSET_ROOT / "models/llama3-8B-instruct"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/contextpilot/online"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("baseline", "contextpilot"))
    parser.add_argument("--topk", required=True, type=int, choices=(1, 3, 5, 10))
    parser.add_argument(
        "--workload", choices=("sequential", "poisson"), default="poisson"
    )
    parser.add_argument("--arrival-rate-rps", type=float, default=24.0)
    parser.add_argument("--arrival-seed", type=int, default=271828)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--window-requests", type=int, default=1000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--prompt-policy",
        choices=("matched", "official-unmatched"),
        default="matched",
        help=(
            "matched uses the same ContextPilot prompt template in both arms; "
            "official-unmatched reproduces the historical prompt-generator A/B."
        ),
    )
    parser.add_argument("--engine-url", default="http://127.0.0.1:30000")
    parser.add_argument("--contextpilot-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--linkage-method", default="average")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def official_revision() -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(OFFICIAL_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    return revision, dirty


def load_inputs(
    topk: int, max_requests: int | None
) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, str]]:
    with NQ_TEST_PATH.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 3610:
        raise RuntimeError(f"Expected 3,610 NQ-test rows, got {len(rows)}")
    if max_requests is not None:
        if max_requests <= 0 or max_requests > len(rows):
            raise ValueError("--max-requests must be in [1, 3610]")
        rows = rows[:max_requests]

    trace = np.load(DOC_IDS_PATH, mmap_mode="r")
    if trace.shape != (3610, 10):
        raise RuntimeError(f"Unexpected retrieval trace shape: {trace.shape}")
    doc_ids = np.asarray(trace[: len(rows), :topk], dtype=np.int64)

    raw_documents = json.loads(DOCUMENT_CACHE.read_text(encoding="utf-8"))
    documents = {int(key): value for key, value in raw_documents.items()}
    needed = {int(value) for value in doc_ids.reshape(-1)}
    missing = needed.difference(documents)
    if missing:
        raise RuntimeError(f"Document cache is missing {len(missing)} retrieved IDs")
    return rows, doc_ids, documents


def poisson_offsets(count: int, rate: float, seed: int) -> np.ndarray:
    if rate <= 0:
        raise ValueError("--arrival-rate-rps must be positive")
    offsets = np.zeros(count, dtype=np.float64)
    if count > 1:
        rng = np.random.default_rng(seed)
        offsets[1:] = np.cumsum(rng.exponential(1.0 / rate, count - 1))
    return offsets


def truncate_prompt(prompt: str, tokenizer, token_budget: int) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= token_budget:
        return prompt, len(token_ids), False
    left = token_budget // 2
    right = token_budget - left
    token_ids = token_ids[:left] + token_ids[-right:]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    final_ids = tokenizer.encode(prompt, add_special_tokens=False)[:token_budget]
    if len(final_ids) != len(token_ids):
        prompt = tokenizer.decode(final_ids, skip_special_tokens=False)
    return prompt, len(final_ids), True


def cached_tokens_from_response(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return prompt_tokens, cached_tokens, completion_tokens


def prometheus_counter(text: str, metric: str) -> int:
    pattern = re.compile(
        rf"^{re.escape(metric)}\{{[^}}]*\}}\s+([0-9.eE+-]+)$", re.MULTILINE
    )
    values = [float(match.group(1)) for match in pattern.finditer(text)]
    if len(values) != 1:
        raise RuntimeError(f"Expected one {metric} series, found {len(values)}")
    return int(round(values[0]))


def system_windows(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for start in range(0, len(records), size):
        rows = records[start : start + size]
        if len(rows) != size:
            break
        prompt = sum(int(row["prompt_tokens"]) for row in rows)
        cached = sum(int(row["cached_tokens"]) for row in rows)
        windows.append(
            {
                "window_index": len(windows),
                "request_start": start,
                "request_end_exclusive": start + size,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "hit_rate": cached / prompt if prompt else 0.0,
            }
        )
    return windows


async def main_async(args: argparse.Namespace) -> None:
    revision, dirty = official_revision()
    if dirty:
        raise RuntimeError(
            "Official ContextPilot checkout is dirty; refusing benchmark"
        )

    rows, doc_ids, documents = load_inputs(args.topk, args.max_requests)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    # Import only the official prompt builders.  Reordering itself is always
    # performed out-of-process by the official /reorder endpoint.
    import sys

    sys.path.insert(0, str(OFFICIAL_ROOT))
    from contextpilot.utils.prompt_generator import (  # noqa: PLC0415
        prompt_generator,
        prompt_generator_baseline,
    )

    if args.workload == "poisson":
        arrivals = poisson_offsets(len(rows), args.arrival_rate_rps, args.arrival_seed)
    else:
        arrivals = np.zeros(len(rows), dtype=np.float64)

    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=512)
    records: list[dict[str, Any] | None] = [None] * len(rows)
    reordered_count = 0
    truncated_count = 0
    start = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(f"{args.engine_url}/v1/models") as response:
            models = await response.json()
            if response.status != 200:
                raise RuntimeError(f"Engine health failed: {response.status} {models}")
            model_name = models["data"][0]["id"]

        async with session.get(f"{args.engine_url}/metrics") as response:
            metrics_before_text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Engine metrics failed: HTTP {response.status}")
        metrics_before = {
            "prompt_tokens": prometheus_counter(
                metrics_before_text, "vllm:prefix_cache_queries_total"
            ),
            "cached_tokens": prometheus_counter(
                metrics_before_text, "vllm:prefix_cache_hits_total"
            ),
        }

        if args.mode == "contextpilot":
            async with session.post(
                f"{args.contextpilot_url}/reset", json={}
            ) as response:
                reset_body = await response.json()
                if response.status != 200:
                    raise RuntimeError(
                        f"ContextPilot reset failed: {response.status} {reset_body}"
                    )

        async def infer(
            request_index: int,
            request_id: str,
            prompt: str,
            target_arrival_s: float,
            actual_arrival_s: float,
            admitted_s: float,
            reorder_s: float,
            reorder_mode: str,
            matched_count: int,
            prompt_truncated: bool,
        ) -> None:
            request_started = time.perf_counter()
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "request_id": request_id,
            }
            async with session.post(
                f"{args.engine_url}/v1/completions", json=payload
            ) as response:
                body = await response.json()
                if response.status != 200:
                    raise RuntimeError(
                        f"Request {request_index} failed: HTTP {response.status}: {body}"
                    )
            prompt_tokens, cached_tokens, completion_tokens = (
                cached_tokens_from_response(body)
            )
            records[request_index] = {
                "request_index": request_index,
                "request_id": request_id,
                "target_arrival_s": target_arrival_s,
                "actual_arrival_s": actual_arrival_s,
                "arrival_lateness_s": actual_arrival_s - target_arrival_s,
                "admitted_s": admitted_s,
                "admission_wait_s": admitted_s - actual_arrival_s,
                "reorder_s": reorder_s,
                "reorder_mode": reorder_mode,
                "matched_count": matched_count,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "request_hit_rate_diagnostic": cached_tokens / prompt_tokens
                if prompt_tokens
                else 0.0,
                "prompt_truncated": int(prompt_truncated),
                "engine_response_s": time.perf_counter() - request_started,
                "e2e_from_target_arrival_s": time.perf_counter()
                - start
                - target_arrival_s,
            }

        admission = asyncio.Semaphore(
            1 if args.workload == "sequential" else args.max_inflight
        )

        async def handle_request(
            request_index: int,
            row: dict[str, Any],
            target_arrival_s: float,
            actual_arrival_s: float,
        ) -> None:
            nonlocal reordered_count, truncated_count
            async with admission:
                admitted_s = time.perf_counter() - start
                original = [int(value) for value in doc_ids[request_index]]
                reordered = original
                reorder_s = 0.0
                reorder_mode = "baseline"
                matched_count = 0
                request_id = f"req-baseline-{request_index:06d}"

                if args.mode == "contextpilot":
                    reorder_started = time.perf_counter()
                    async with session.post(
                        f"{args.contextpilot_url}/reorder",
                        json={
                            "contexts": [original],
                            "initial_tokens_per_context": 0,
                            "alpha": args.alpha,
                            "use_gpu": False,
                            "linkage_method": args.linkage_method,
                        },
                    ) as response:
                        reorder_body = await response.json()
                        if response.status != 200:
                            raise RuntimeError(
                                f"/reorder failed at request {request_index}: "
                                f"HTTP {response.status}: {reorder_body}"
                            )
                    reorder_s = time.perf_counter() - reorder_started
                    reorder_mode = str(reorder_body.get("mode") or "unknown")
                    matched_count = int(reorder_body.get("matched_count") or 0)
                    reordered = [
                        int(value) for value in reorder_body["reordered_contexts"][0]
                    ]
                    request_id = str(reorder_body["request_ids"][0])
                    if sorted(reordered) != sorted(original):
                        raise RuntimeError(
                            f"ContextPilot changed the document set at {request_index}"
                        )
                    reordered_count += int(reordered != original)

                answer = row.get("golden_answers") or row.get("answer") or []
                if args.prompt_policy == "matched" or args.mode == "contextpilot":
                    item = {
                        "qid": request_index,
                        "answer": answer,
                        "question": row["question"],
                        "top_k_doc_id": (
                            reordered if args.mode == "contextpilot" else original
                        ),
                        "orig_top_k_doc_id": original,
                    }
                    prompts, _, _ = prompt_generator(
                        documents,
                        [item],
                        tokenizer=tokenizer,
                        apply_template=True,
                    )
                else:
                    item = {
                        "qid": request_index,
                        "answer": answer,
                        "text": row["question"],
                        "top_k_doc_id": original,
                    }
                    prompts, _, _ = prompt_generator_baseline(
                        documents,
                        [item],
                        tokenizer=tokenizer,
                        apply_template=True,
                    )

                # Reserve a small tokenizer round-trip margin.  Llama-3's
                # decode/re-encode path can add one boundary token relative to
                # the client count, while vLLM enforces input+output <= 2048.
                prompt, _, was_truncated = truncate_prompt(
                    prompts[0],
                    tokenizer,
                    args.max_model_len - args.max_tokens - 8,
                )
                truncated_count += int(was_truncated)
                await infer(
                    request_index,
                    request_id,
                    prompt,
                    target_arrival_s,
                    actual_arrival_s,
                    admitted_s,
                    reorder_s,
                    reorder_mode,
                    matched_count,
                    was_truncated,
                )

        tasks: list[asyncio.Task[None]] = []
        for request_index, row in enumerate(rows):
            if args.workload == "poisson":
                delay = start + float(arrivals[request_index]) - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            actual_arrival_s = time.perf_counter() - start
            task = asyncio.create_task(
                handle_request(
                    request_index,
                    row,
                    float(arrivals[request_index]),
                    actual_arrival_s,
                )
            )
            tasks.append(task)
            if args.workload == "sequential":
                await task
            if (request_index + 1) % 250 == 0:
                complete = sum(record is not None for record in records)
                print(
                    f"{args.mode} topk={args.topk}: arrived {request_index + 1:,}/"
                    f"{len(rows):,}, completed {complete:,}",
                    flush=True,
                )

        await asyncio.gather(*tasks)

        async with session.get(f"{args.engine_url}/metrics") as response:
            metrics_after_text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Engine metrics failed: HTTP {response.status}")
        metrics_after = {
            "prompt_tokens": prometheus_counter(
                metrics_after_text, "vllm:prefix_cache_queries_total"
            ),
            "cached_tokens": prometheus_counter(
                metrics_after_text, "vllm:prefix_cache_hits_total"
            ),
        }

    complete = [record for record in records if record is not None]
    if len(complete) != len(rows):
        raise RuntimeError(f"Only {len(complete)}/{len(rows)} requests completed")
    prompt_total = sum(int(row["prompt_tokens"]) for row in complete)
    cached_total = sum(int(row["cached_tokens"]) for row in complete)
    metrics_prompt_delta = (
        metrics_after["prompt_tokens"] - metrics_before["prompt_tokens"]
    )
    metrics_cached_delta = (
        metrics_after["cached_tokens"] - metrics_before["cached_tokens"]
    )
    if metrics_prompt_delta != prompt_total or metrics_cached_delta != cached_total:
        raise RuntimeError(
            "OpenAI usage does not match native vLLM counters: "
            f"usage=({prompt_total}, {cached_total}), "
            f"metrics=({metrics_prompt_delta}, {metrics_cached_delta})"
        )
    windows = system_windows(complete, args.window_requests)
    window_rates = [float(window["hit_rate"]) for window in windows]
    point_dir = args.output_root / args.workload / f"{args.mode}_topk{args.topk}"
    write_csv(point_dir / "requests.csv", complete)
    write_csv(point_dir / "system_hit_rate_windows.csv", windows)

    import vllm  # noqa: PLC0415

    summary = {
        "status": "complete",
        "system": "ContextPilot"
        if args.mode == "contextpilot"
        else "vLLM LRU baseline",
        "implementation": "official ContextPilot HTTP online path",
        "contextpilot_repository": "https://github.com/EfficientContext/ContextPilot.git",
        "contextpilot_tag": "v0.5.0",
        "contextpilot_commit": revision,
        "contextpilot_git_dirty": dirty,
        "vllm_version": vllm.__version__,
        "dataset": "NQ test",
        "request_count": len(rows),
        "topk": args.topk,
        "retrieval_nprobe": 128,
        "model": str(MODEL_PATH),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "prompt_token_safety_reserve": 8,
        "load_generator_max_inflight": (
            1 if args.workload == "sequential" else args.max_inflight
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prompt_policy": args.prompt_policy,
        "enable_prefix_caching": True,
        "max_num_seqs": 32,
        "workload": args.workload,
        "arrival_process": (
            "next request after prior response (official online A/B convention)"
            if args.workload == "sequential"
            else "Poisson exponential inter-arrivals; source order retained"
        ),
        "arrival_rate_rps": args.arrival_rate_rps
        if args.workload == "poisson"
        else None,
        "arrival_seed": args.arrival_seed if args.workload == "poisson" else None,
        "aggregate_prompt_tokens": prompt_total,
        "aggregate_cached_tokens": cached_total,
        "aggregate_hit_rate": cached_total / prompt_total if prompt_total else 0.0,
        "native_vllm_prompt_tokens_delta": metrics_prompt_delta,
        "native_vllm_cached_tokens_delta": metrics_cached_delta,
        "native_vllm_counter_crosscheck": "exact match",
        "system_window_request_count": args.window_requests,
        "system_window_count": len(windows),
        "average_system_window_hit_rate": statistics.fmean(window_rates)
        if window_rates
        else None,
        "max_system_window_hit_rate": max(window_rates) if window_rates else None,
        "max_definition": "maximum token-weighted hit rate over complete non-overlapping system windows; not a per-request maximum",
        "requests_with_any_hit": sum(int(row["cached_tokens"]) > 0 for row in complete),
        "reordered_request_count": reordered_count,
        "reordered_request_rate": reordered_count / len(rows),
        "contextpilot_matched_request_count": sum(
            int(row["matched_count"] > 0) for row in complete
        ),
        "contextpilot_matched_request_rate": sum(
            int(row["matched_count"] > 0) for row in complete
        )
        / len(rows),
        "contextpilot_reorder_modes": {
            mode: sum(int(row["reorder_mode"] == mode) for row in complete)
            for mode in sorted({str(row["reorder_mode"]) for row in complete})
        },
        "prompt_trace_sha256": hashlib.sha256(
            "".join(str(row["prompt_sha256"]) for row in complete).encode("ascii")
        ).hexdigest(),
        "prompt_truncated_count": truncated_count,
        "wall_time_s": time.perf_counter() - start,
        "mean_engine_response_s": statistics.fmean(
            float(row["engine_response_s"]) for row in complete
        ),
        "p99_engine_response_s": float(
            np.percentile([row["engine_response_s"] for row in complete], 99)
        ),
        "mean_admission_wait_s": statistics.fmean(
            float(row["admission_wait_s"]) for row in complete
        ),
        "p99_admission_wait_s": float(
            np.percentile([row["admission_wait_s"] for row in complete], 99)
        ),
        "mean_reorder_s": statistics.fmean(float(row["reorder_s"]) for row in complete),
        "python_version": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(point_dir / "summary.json", summary)
    print("POINT_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
