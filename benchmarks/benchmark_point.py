#!/usr/bin/env python3
"""Run one isolated MnemoRAG evaluation point on the 21M Wikipedia corpus.

Each invocation creates exactly one retrieval stack and one AsyncLLM engine.
Campaign drivers must launch a fresh process for every configuration so that
the query pool and vLLM prefix cache never leak across experimental points.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

# Resource preflight may initialize CUDA before vLLM creates its engine core.
# Spawn prevents the child from inheriting an initialized CUDA runtime.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

try:
    from .experiment_validation import (
        CANONICAL_SYSTEM_PROMPT,
        CANONICAL_USER_PROMPT,
        DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS,
        expected_prompt_contract,
    )
    from .fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        build_trace_manifest,
        load_trace_bundle,
        load_trace_sample_indices,
        summarize_latency_slo,
        validate_trace_workload,
    )
    from .vllm_contract import validate_effective_vllm_contract
except ImportError:
    from experiment_validation import (
        CANONICAL_SYSTEM_PROMPT,
        CANONICAL_USER_PROMPT,
        DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS,
        expected_prompt_contract,
    )
    from fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        build_trace_manifest,
        load_trace_bundle,
        load_trace_sample_indices,
        summarize_latency_slo,
        validate_trace_workload,
    )
    from vllm_contract import validate_effective_vllm_contract

try:
    from .runtime_resource_monitor import (
        RuntimeResourceMonitor,
        physical_sibling_cpu_ids,
    )
except ImportError:
    from runtime_resource_monitor import (
        RuntimeResourceMonitor,
        physical_sibling_cpu_ids,
    )


PROJECT_ROOT = Path(
    os.environ.get("MNEMORAG_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
MNEMORAG_VLLM_ROOT = Path(
    os.environ.get(
        "MNEMORAG_VLLM_ROOT",
        PROJECT_ROOT / "third_party/vllm",
    )
).resolve()
DATASET_PATHS = {
    "nq": ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    "hotpotqa": ASSET_ROOT / "qa/datasets/hotpotqa/train.jsonl",
    "webquestions": ASSET_ROOT / "qa/datasets/web_questions/train.jsonl",
    "squad": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
    "2wikimultihopqa": ASSET_ROOT / "qa/datasets/2wikimultihopqa/train.jsonl",
}
COMPONENTS = {
    "vanilla": (False, False, False, "lru"),
    "async_baseline": (False, False, True, "lru"),
    # Full MnemoRAG cache/scheduling stack with exact Full-IVF retrieval.
    # This mode isolates Prefix-KV effects from similarity-aware retrieval.
    "kv_full": (False, True, True, "heat"),
    "sar": (True, False, False, "lru"),
    "hotness": (False, True, False, "heat"),
    "async": (False, False, True, "lru"),
    "sar_hotness": (True, True, False, "heat"),
    # Isolation modes used only by the formal component ablation.
    "full_no_hotness": (True, False, True, "lru"),
    "full_no_async": (True, True, False, "heat"),
    "full": (True, True, True, "heat"),
}


def validate_single_query_probe_parallelism_contract(
    system_label: str, retrieval_profile: dict | None
) -> dict:
    """Keep MnemoRAG intra-query IVF parallelism out of FlashRAG."""
    if retrieval_profile is None:
        raise RuntimeError("retrieval profile is required for IVF audit")
    profile = retrieval_profile.get("single_query_probe_parallelism")
    if not isinstance(profile, dict):
        raise RuntimeError("single-query IVF parallelism evidence is missing")

    expected_enabled = system_label == "mnemorag"
    if bool(profile.get("enabled")) != expected_enabled:
        raise RuntimeError(
            "single-query IVF parallelism system contract mismatch: "
            f"system={system_label}, profile={profile}"
        )
    if system_label == "flashrag" and (
        int(profile.get("searches", 0)) != 0
        or int(profile.get("maximum_partitions", 1)) != 1
    ):
        raise RuntimeError(
            "FlashRAG must use standard Faiss search without intra-query "
            f"IVF partitioning: {profile}"
        )
    if system_label == "mnemorag" and int(profile.get("maximum_partitions", 0)) != 8:
        raise RuntimeError(
            f"MnemoRAG must expose the eight-partition IVF limit: {profile}"
        )
    return profile


DEFAULT_CPU_KV_CACHE_GB = 256.0
DEFAULT_CPU_KV_CACHE_GIB = DEFAULT_CPU_KV_CACHE_GB * 1000**3 / 1024**3
DEFAULT_CPU_KV_BLOCK_SIZE = 16
REQUIRED_PHYSICAL_CPU_CORES = 64
REQUIRED_FAISS_SEARCH_WORKERS = 64
REQUIRED_FAISS_OMP_THREADS = 1
MIN_GPU_RESERVATION_HANDOFF_MIB = 128.0


def _hold_startup_gpu_reservation(
    memory_mib: float,
    ready_connection,
    release_event,
) -> None:
    """Occupy the selected GPU while the CPU-side index is loading."""
    try:
        import torch

        byte_count = max(1, int(memory_mib * 1024 * 1024))
        reservation = torch.empty(
            byte_count,
            dtype=torch.uint8,
            device="cuda:0",
        )
        reservation.zero_()
        torch.cuda.synchronize()
        ready_connection.send(
            {
                "status": "ready",
                "pid": os.getpid(),
                "allocated_bytes": byte_count,
            }
        )
        release_event.wait()
        del reservation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except BaseException as error:  # pragma: no cover - host diagnostic
        try:
            ready_connection.send(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        ready_connection.close()


def start_startup_gpu_reservation(memory_mib: float) -> dict | None:
    if memory_mib <= 0:
        return None
    context = multiprocessing.get_context("spawn")
    ready_parent, ready_child = context.Pipe(duplex=False)
    release_event = context.Event()
    process = context.Process(
        target=_hold_startup_gpu_reservation,
        args=(memory_mib, ready_child, release_event),
        name="MnemoRAG-GPUReservation",
    )
    process.start()
    ready_child.close()
    if not ready_parent.poll(60):
        process.terminate()
        process.join(timeout=10)
        raise RuntimeError("Timed out while reserving the startup GPU")
    report = ready_parent.recv()
    ready_parent.close()
    if report.get("status") != "ready":
        process.join(timeout=10)
        raise RuntimeError(f"Unable to reserve the startup GPU: {report}")
    report.update(
        {
            "requested_mib": float(memory_mib),
            "released": False,
            "release_exitcode": None,
        }
    )
    print(
        "[MnemoRAG] Startup GPU reservation: " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return {
        "process": process,
        "release_event": release_event,
        "report": report,
    }


def release_startup_gpu_reservation(handle: dict | None) -> dict | None:
    if handle is None:
        return None
    report = handle["report"]
    if report.get("released"):
        return report
    process = handle["process"]
    handle["release_event"].set()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
    report["released"] = not process.is_alive()
    report["release_exitcode"] = process.exitcode
    if not report["released"] or process.exitcode != 0:
        raise RuntimeError(f"Startup GPU reservation did not exit cleanly: {report}")
    print(
        "[MnemoRAG] Startup GPU reservation released: "
        + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return report


def release_loaded_input_page_cache(paths: list[Path]) -> dict[str, object]:
    """Release clean file pages after eager loaders copied their contents."""
    started = time.perf_counter()
    records = []
    status = "complete"
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {
            "status": "unsupported",
            "elapsed_s": time.perf_counter() - started,
            "files": [],
        }
    for path in paths:
        record = {"path": str(path), "size_bytes": None, "status": "complete"}
        try:
            record["size_bytes"] = path.stat().st_size
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.posix_fadvise(
                    descriptor,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )
            finally:
                os.close(descriptor)
        except Exception as error:  # pragma: no cover - host diagnostic
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            status = "error"
        records.append(record)
    return {
        "status": status,
        "elapsed_s": time.perf_counter() - started,
        "files": records,
    }


def build_prompt_template(config, order_mode: str, prompt_template_cls):
    """Build the canonical QA template and apply the ordering-hint policy."""
    if order_mode not in {"original", "hotness", "hotness_hint"}:
        raise ValueError(f"Unsupported order mode: {order_mode}")
    prompt_template = prompt_template_cls(
        config,
        system_prompt=CANONICAL_SYSTEM_PROMPT,
        user_prompt=CANONICAL_USER_PROMPT,
    )
    if order_mode != "hotness_hint":
        original_get_string = prompt_template.get_string

        def get_string_without_hint(*positional, **keywords):
            keywords.pop("original_retrieval_result", None)
            return original_get_string(*positional, **keywords)

        prompt_template.get_string = get_string_without_hint
    return prompt_template


def resolve_similarity_nprobe_bounds(
    nprobe: int,
    minimum_fraction: float | None,
    maximum_fraction: float | None,
) -> tuple[int | None, int | None]:
    """Resolve optional similarity-aware probe fractions into integer budgets."""
    if nprobe <= 0:
        raise ValueError("nprobe must be positive")

    def resolve(value: float | None, name: str) -> int | None:
        if value is None:
            return None
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        return min(nprobe, max(1, math.ceil(nprobe * value)))

    minimum = resolve(minimum_fraction, "similarity nprobe minimum fraction")
    maximum = resolve(maximum_fraction, "similarity nprobe maximum fraction")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("similarity nprobe minimum must not exceed maximum")
    return minimum, maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-label", choices=("mnemorag", "flashrag"), default="mnemorag"
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="nq")
    parser.add_argument(
        "--workflow", choices=("onepass", "recomp", "irg"), default="onepass"
    )
    parser.add_argument("--components", choices=sorted(COMPONENTS), default="full")
    parser.add_argument(
        "--order-mode",
        choices=("original", "hotness", "hotness_hint"),
        default="hotness_hint",
    )
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument(
        "--request-cycle-size",
        type=int,
        default=0,
        help=(
            "Cycle the first N sampled requests to create a controlled "
            "temporal-reuse trace; zero keeps all sampled requests unique."
        ),
    )
    parser.add_argument("--arrival-rate", type=float, default=24.0)
    parser.add_argument(
        "--arrival-burstiness",
        type=float,
        default=1.0,
        help=(
            "Gamma arrival-process shape. The vLLM-compatible default 1.0 "
            "is Poisson; larger values approach constant inter-arrival times."
        ),
    )
    parser.add_argument(
        "--normalize-arrival-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rescale each finite generated trace to exactly N / QPS seconds, "
            "matching vLLM online-serving benchmark semantics."
        ),
    )
    parser.add_argument(
        "--arrival-trace",
        type=Path,
        default=None,
        help="Canonical .npz arrival trace shared by every compared system.",
    )
    parser.add_argument(
        "--online-warmup-indices",
        type=Path,
        default=None,
        help=(
            "Disjoint source-row indices executed on the same engine before "
            "measurement. Warmup requests preserve cache/query state but are "
            "excluded from latency, quality, prefix-token, and SLO metrics."
        ),
    )
    parser.add_argument("--latency-slo-seconds", type=float, default=10.0)
    parser.add_argument("--campaign-group-id", default="standalone")
    parser.add_argument("--campaign-repeat", type=int, default=0)
    parser.add_argument("--campaign-order-position", type=int, default=0)
    parser.add_argument("--campaign-variant", default="full")
    parser.add_argument("--topk", type=int, choices=(1, 3, 5, 10), default=3)
    parser.add_argument("--nprobe", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument(
        "--cache-aware-prefill-batching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep CPU-KV-ready and cold prompts in separate generation "
            "batches without waiting for future arrivals."
        ),
    )
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument(
        "--async-retrieval-policy",
        choices=(
            "prefill_start",
            "prefill_batch",
            "stage_aware",
            "adaptive",
            "sequential",
            "fixed_delay",
            "eager",
        ),
        default="prefill_batch",
    )
    parser.add_argument(
        "--cross-microbatch-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep one look-ahead workflow slot so retrieval for the next "
            "outer microbatch can overlap the current decode stage."
        ),
    )
    parser.add_argument(
        "--direct-vllm-dispatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit each retrieval-ready request directly to vLLM continuous "
            "batching without an application-side generation-stage gate."
        ),
    )
    parser.add_argument("--async-fixed-delay-ms", type=float, default=60.0)
    parser.add_argument("--async-adaptive-ema-alpha", type=float, default=0.25)
    parser.add_argument(
        "--async-adaptive-max-prefill-overlap-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--async-adaptive-safety-margin-ms", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force max_tokens for controlled decode-window experiments.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help=(
            "vLLM scheduler token budget per engine step. Zero keeps the "
            "vLLM default; positive values are passed explicitly."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        default="auto",
        help="vLLM GPU KV-cache dtype. Figure 14 uses fp8.",
    )
    parser.add_argument(
        "--startup-gpu-reservation-mib",
        type=float,
        default=0.0,
        help=(
            "Temporarily reserve target-GPU memory while the CPU-side index "
            "loads, then hand ownership to the retriever before vLLM starts."
        ),
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM prefix caching. Use --no-prefix-caching only for a "
            "controlled cache-causality ablation with CPU KV offload disabled."
        ),
    )
    parser.add_argument(
        "--kv-policy-override",
        choices=("heat", "lru"),
        default=None,
        help="Override only the GPU prefix-KV eviction policy for ablations.",
    )
    parser.add_argument(
        "--kv-heat-half-life-s",
        type=float,
        default=600.0,
        help="Half-life for the exponentially decayed heat policy.",
    )
    parser.add_argument(
        "--kv-heat-formula",
        choices=("exponential_decay", "legacy_log_denominator"),
        default="exponential_decay",
        help=(
            "GPU KV heat formula. legacy_log_denominator is the repaired "
            "historical H_i/log(1+t_i/tau) timestamp implementation with a "
            "zero-time guard and explicit time scale."
        ),
    )
    parser.add_argument(
        "--kv-profile-priority-mode",
        choices=("formula_tiebreak", "profile_only"),
        default="formula_tiebreak",
        help=(
            "How an offline next-use profile composes with heat. The default "
            "keeps next-use as the primary key and executes the selected heat "
            "formula as a bounded tie-break; profile_only is an audited ablation."
        ),
    )
    parser.add_argument(
        "--kv-heat-legacy-time-scale-s",
        type=float,
        default=1.0,
        help=(
            "Time unit used by the repaired legacy logarithmic denominator; "
            "must be positive so the log argument is dimensionless."
        ),
    )
    parser.add_argument(
        "--kv-heat-source",
        choices=("document", "prefix_frequency", "hybrid"),
        default="document",
        help="Signal used as H_i by the heat-aware GPU KV policy.",
    )
    parser.add_argument(
        "--document-kv-heat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach document access heat to prompt token intervals.",
    )
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--faiss-omp-threads", type=int, default=1)
    parser.add_argument("--faiss-search-workers", type=int, default=64)
    parser.add_argument(
        "--faiss-mmap-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Memory-map IVF inverted lists instead of eagerly loading them.",
    )
    parser.add_argument(
        "--drop-loaded-input-page-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After eager index/corpus loading, advise Linux to evict their "
            "clean file pages. The in-memory Faiss index and corpus remain "
            "unchanged; this avoids retaining both source pages and copies."
        ),
    )
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-preflight-cpu-busy-percent", type=float, default=20.0)
    parser.add_argument(
        "--max-preflight-cpu-max-busy-percent", type=float, default=50.0
    )
    parser.add_argument("--max-preflight-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-preflight-gpu-util-percent", type=float, default=10.0)
    parser.add_argument("--max-preflight-load-per-cpu", type=float, default=0.10)
    parser.add_argument("--runtime-resource-sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-runtime-external-cpu-busy-percent",
        type=float,
        default=5.0,
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--tau-reuse", type=float, default=0.95)
    parser.add_argument("--tau-guided", type=float, default=0.85)
    parser.add_argument(
        "--similarity-nprobe-min-fraction",
        type=float,
        default=None,
        help=(
            "Optional similarity-aware minimum query-list budget as a "
            "fraction of nprobe. "
            "The resolved integer is recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--similarity-nprobe-max-fraction",
        type=float,
        default=None,
        help=(
            "Optional similarity-aware maximum query-list budget as a "
            "fraction of nprobe. "
            "The resolved integer is recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--similarity-nprobe-gamma",
        type=float,
        default=None,
        help="Optional exponent for the similarity-aware probe-budget curve.",
    )
    parser.add_argument(
        "--similarity-nprobe-curve",
        choices=("power_decay", "late_decay"),
        default=None,
        help=(
            "Optional similarity-aware probe-budget curve; omitted keeps "
            "retriever defaults."
        ),
    )
    parser.add_argument(
        "--query-base-exact-lookup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preassigned-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use search_preassigned for similarity-guided IVF lists. "
            "Disable only for the nprobe-only ablation."
        ),
    )
    parser.add_argument("--irg-rounds", type=int, default=2)
    parser.add_argument(
        "--cpu-kv-cache-gib",
        type=float,
        default=DEFAULT_CPU_KV_CACHE_GIB,
        help=(
            "Opt-in host-memory capacity for MnemoRAG KV offload. The runner "
            "derives the vLLM block count from this size and the model config; "
            "zero disables offload. The default is exactly 256 decimal GB "
            "(about 238.42 GiB)."
        ),
    )
    parser.add_argument(
        "--cpu-kv-num-blocks",
        type=int,
        default=0,
        help=(
            "Exact CPU KV capacity in blocks. A positive value overrides the "
            "capacity-derived block count; Figure 14 uses 130000."
        ),
    )
    parser.add_argument(
        "--cpu-kv-block-size",
        type=int,
        default=DEFAULT_CPU_KV_BLOCK_SIZE,
        help="Number of tokens in one offloaded KV block.",
    )
    parser.add_argument(
        "--cpu-kv-eviction-policy",
        choices=("lru", "arc", "heat"),
        default="heat",
    )
    parser.add_argument(
        "--cpu-kv-batch-h2d-transfers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cpu-kv-batch-d2h-transfers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cpu-kv-h2d-max-group-requests",
        type=int,
        default=0,
        help=(
            "Maximum requests merged into one H2D completion job. Zero keeps "
            "the legacy one-job-per-scheduler-step behavior."
        ),
    )
    parser.add_argument(
        "--cpu-kv-h2d-scheduling",
        choices=("async", "sync", "adaptive"),
        default="async",
        help=(
            "CPU-KV reload scheduling. Async is the validated default. "
            "Adaptive overlaps decode-only steps but moves H2D after "
            "prefill-sensitive forwards, which may move memory-bandwidth "
            "contention onto CPU retrieval."
        ),
    )
    parser.add_argument(
        "--cpu-kv-document-prefix-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Offload only complete KV blocks through the final annotated "
            "document token. This excludes request-specific query and decode "
            "tails from CPU-KV admission."
        ),
    )
    parser.add_argument(
        "--cpu-kv-min-document-heat-to-store",
        type=float,
        default=1.0,
        help=(
            "Minimum annotated document heat required before admitting a "
            "request prefix to CPU KV. Values above one suppress one-off "
            "D2H writes."
        ),
    )
    parser.add_argument(
        "--cpu-kv-h2d-stream-priority",
        type=int,
        default=0,
        help="CUDA stream priority for CPU-to-GPU KV reloads.",
    )
    parser.add_argument(
        "--cpu-kv-d2h-stream-priority",
        type=int,
        default=0,
        help="CUDA stream priority for GPU-to-CPU KV offloads.",
    )
    parser.add_argument(
        "--cpu-kv-host-memory-mode",
        choices=("pinned_eager", "pageable_lazy", "hybrid_pinned_lazy"),
        default="hybrid_pinned_lazy",
        help=(
            "Host allocation for the logical CPU KV capacity. pageable_lazy "
            "commits pages only when blocks are written, avoiding an eager "
            "full-capacity pinned allocation; pinned_eager preserves the "
            "legacy behavior."
        ),
    )
    parser.add_argument(
        "--cpu-kv-pinned-tier-gib",
        type=float,
        default=192_000_000_000 / 1024**3,
        help=(
            "Pinned tier within hybrid_pinned_lazy capacity. The default is "
            "exactly 192 decimal GB; the remainder stays lazily committed."
        ),
    )
    parser.add_argument(
        "--cpu-kv-lazy-mmap-dir",
        type=Path,
        default=Path("/dev/shm"),
        help=(
            "Backing directory for the lazy tier in hybrid_pinned_lazy mode. "
            "Use a disk-backed filesystem when the logical CPU KV capacity "
            "would otherwise create unreclaimable tmpfs pressure."
        ),
    )
    parser.add_argument(
        "--cpu-kv-staging-buffer-blocks",
        type=int,
        default=1024,
        help=(
            "Reusable pinned staging capacity per transfer direction. "
            "Must cover the largest H2D or D2H transfer job."
        ),
    )
    parser.add_argument(
        "--require-cpu-kv-h2d-samples",
        action="store_true",
        help="Fail when CPU KV offload is enabled but no H2D reload completes.",
    )
    parser.add_argument(
        "--require-cpu-kv-d2h-samples",
        action="store_true",
        help="Fail when CPU KV offload is enabled but no D2H offload completes.",
    )
    parser.add_argument(
        "--record-stage-breakdown",
        action="store_true",
        help="Record retrieval, prefill, and decode service latencies for Figure 15.",
    )
    parser.add_argument(
        "--record-prefix-distribution",
        action="store_true",
        help="Record per-request prefix hit rates for Figure 14 mean/max panels.",
    )
    parser.add_argument(
        "--retrieval-reference-csv",
        type=Path,
        default=None,
        help=(
            "Replay retrieval IDs and scores from a paired Base requests.csv "
            "after executing and timing the real retriever."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation",
    )
    return parser.parse_args()


def derive_cpu_kv_layout(
    args: argparse.Namespace,
) -> dict[str, int | float | bool | str]:
    if args.cpu_kv_num_blocks < 0:
        raise ValueError("--cpu-kv-num-blocks must be nonnegative")
    enabled = args.cpu_kv_num_blocks > 0 or args.cpu_kv_cache_gib > 0
    if not enabled:
        return {
            "enabled": False,
            "capacity_gb": 0.0,
            "capacity_gib": 0.0,
            "block_size_tokens": args.cpu_kv_block_size,
            "block_bytes": 0,
            "kv_cache_dtype": str(getattr(args, "kv_cache_dtype", "auto")),
            "dtype_bytes": 0,
            "num_cpu_blocks": 0,
            "capacity_source": "disabled",
            "eviction_policy": args.cpu_kv_eviction_policy,
            "batch_h2d_transfers": bool(args.cpu_kv_batch_h2d_transfers),
            "batch_d2h_transfers": bool(args.cpu_kv_batch_d2h_transfers),
            "h2d_max_group_requests": int(args.cpu_kv_h2d_max_group_requests),
            "h2d_scheduling": args.cpu_kv_h2d_scheduling,
            "document_prefix_only": bool(args.cpu_kv_document_prefix_only),
            "min_document_heat_to_store": float(args.cpu_kv_min_document_heat_to_store),
            "h2d_stream_priority": int(args.cpu_kv_h2d_stream_priority),
            "d2h_stream_priority": int(args.cpu_kv_d2h_stream_priority),
            "host_memory_mode": args.cpu_kv_host_memory_mode,
            "lazy_allocation": None,
            "lazy_mmap_dir": None,
            "lazy_backing_required_bytes": 0,
            "lazy_backing_free_bytes": None,
            "pinned_tier_gb": 0.0,
            "pinned_tier_gib": 0.0,
            "pinned_tier_blocks": 0,
            "staging_buffer_blocks": int(args.cpu_kv_staging_buffer_blocks),
        }
    if args.cpu_kv_block_size < 1:
        raise ValueError("--cpu-kv-block-size must be positive")
    if args.cpu_kv_h2d_max_group_requests < 0:
        raise ValueError("--cpu-kv-h2d-max-group-requests must be nonnegative")
    if args.cpu_kv_min_document_heat_to_store < 1:
        raise ValueError("--cpu-kv-min-document-heat-to-store must be at least one")
    if args.cpu_kv_pinned_tier_gib < 0:
        raise ValueError("--cpu-kv-pinned-tier-gib must be nonnegative")
    if (
        not args.cpu_kv_num_blocks
        and args.cpu_kv_pinned_tier_gib > args.cpu_kv_cache_gib
    ):
        raise ValueError("--cpu-kv-pinned-tier-gib cannot exceed --cpu-kv-cache-gib")
    if args.cpu_kv_staging_buffer_blocks < 1:
        raise ValueError("--cpu-kv-staging-buffer-blocks must be positive")
    model_config = json.loads(
        (ASSET_ROOT / "models/llama3-8B-instruct/config.json").read_text(
            encoding="utf-8"
        )
    )
    hidden_size = int(model_config["hidden_size"])
    attention_heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config.get("num_key_value_heads", attention_heads))
    layers = int(model_config["num_hidden_layers"])
    head_dim = int(model_config.get("head_dim") or hidden_size // attention_heads)
    configured_kv_dtype = str(getattr(args, "kv_cache_dtype", "auto")).lower()
    dtype_name = (
        str(model_config.get("torch_dtype", "bfloat16")).lower()
        if configured_kv_dtype == "auto"
        else configured_kv_dtype
    )
    dtype_bytes = {
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "float32": 4,
        "fp8": 1,
        "fp8_e4m3": 1,
        "fp8_e4m3fn": 1,
        "fp8_e5m2": 1,
    }.get(dtype_name)
    if dtype_bytes is None:
        raise ValueError(f"Unsupported KV dtype for capacity derivation: {dtype_name}")
    block_bytes = (
        2 * layers * kv_heads * head_dim * args.cpu_kv_block_size * dtype_bytes
    )
    if args.cpu_kv_num_blocks:
        num_cpu_blocks = int(args.cpu_kv_num_blocks)
        capacity_bytes = num_cpu_blocks * block_bytes
    else:
        capacity_bytes = int(args.cpu_kv_cache_gib * 1024**3)
        num_cpu_blocks = capacity_bytes // block_bytes
    pinned_tier_bytes = int(args.cpu_kv_pinned_tier_gib * 1024**3)
    if args.cpu_kv_host_memory_mode == "pinned_eager":
        pinned_tier_bytes = capacity_bytes
    elif args.cpu_kv_host_memory_mode == "pageable_lazy":
        pinned_tier_bytes = 0
    pinned_tier_blocks = min(num_cpu_blocks, pinned_tier_bytes // block_bytes)
    if num_cpu_blocks < 1:
        raise ValueError("--cpu-kv-cache-gib is smaller than one KV block")
    lazy_mmap_dir = Path(
        getattr(args, "cpu_kv_lazy_mmap_dir", Path("/dev/shm"))
    )
    if args.cpu_kv_host_memory_mode == "hybrid_pinned_lazy":
        if not lazy_mmap_dir.is_dir():
            raise ValueError(
                "--cpu-kv-lazy-mmap-dir must name an existing directory in "
                "hybrid_pinned_lazy mode"
            )
        lazy_allocation = (
            "tmpfs_mmap"
            if lazy_mmap_dir.resolve() == Path("/dev/shm").resolve()
            else "file_mmap"
        )
        lazy_backing_required_bytes = (
            num_cpu_blocks - pinned_tier_blocks
        ) * block_bytes
        lazy_backing_free_bytes = shutil.disk_usage(lazy_mmap_dir).free
        if (
            lazy_allocation == "file_mmap"
            and lazy_backing_free_bytes < lazy_backing_required_bytes
        ):
            raise ValueError(
                "--cpu-kv-lazy-mmap-dir has insufficient free space: "
                f"requires {lazy_backing_required_bytes} bytes, has "
                f"{lazy_backing_free_bytes} bytes"
            )
    else:
        lazy_allocation = None
        lazy_backing_required_bytes = 0
        lazy_backing_free_bytes = None
    return {
        "enabled": True,
        "capacity_gb": float(capacity_bytes / 1000**3),
        "capacity_gib": float(capacity_bytes / 1024**3),
        "block_size_tokens": int(args.cpu_kv_block_size),
        "block_bytes": int(block_bytes),
        "kv_cache_dtype": dtype_name,
        "dtype_bytes": int(dtype_bytes),
        "num_cpu_blocks": int(num_cpu_blocks),
        "capacity_source": (
            "exact_block_count" if args.cpu_kv_num_blocks else "capacity_gib"
        ),
        "eviction_policy": args.cpu_kv_eviction_policy,
        "batch_h2d_transfers": bool(args.cpu_kv_batch_h2d_transfers),
        "batch_d2h_transfers": bool(args.cpu_kv_batch_d2h_transfers),
        "h2d_max_group_requests": int(args.cpu_kv_h2d_max_group_requests),
        "h2d_scheduling": args.cpu_kv_h2d_scheduling,
        "document_prefix_only": bool(args.cpu_kv_document_prefix_only),
        "min_document_heat_to_store": float(args.cpu_kv_min_document_heat_to_store),
        "h2d_stream_priority": int(args.cpu_kv_h2d_stream_priority),
        "d2h_stream_priority": int(args.cpu_kv_d2h_stream_priority),
        "host_memory_mode": args.cpu_kv_host_memory_mode,
        "lazy_allocation": lazy_allocation,
        "lazy_mmap_dir": (
            str(lazy_mmap_dir)
            if args.cpu_kv_host_memory_mode == "hybrid_pinned_lazy"
            else None
        ),
        "lazy_backing_required_bytes": int(lazy_backing_required_bytes),
        "lazy_backing_free_bytes": (
            int(lazy_backing_free_bytes)
            if lazy_backing_free_bytes is not None
            else None
        ),
        "pinned_tier_gb": float(pinned_tier_blocks * block_bytes / 1000**3),
        "pinned_tier_gib": float(pinned_tier_blocks * block_bytes / 1024**3),
        "pinned_tier_blocks": int(pinned_tier_blocks),
        "staging_buffer_blocks": int(args.cpu_kv_staging_buffer_blocks),
    }


def _read_cpu_counters(cpu_ids: list[int]) -> dict[int, tuple[int, int]]:
    selected = set(cpu_ids)
    counters = {}
    with open("/proc/stat", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu"):
                continue
            suffix = fields[0][3:]
            if not suffix.isdigit() or int(suffix) not in selected:
                continue
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            counters[int(suffix)] = (sum(values), idle)
    return counters


def resource_preflight(args: argparse.Namespace, cpu_ids: list[int]) -> dict:
    monitored_cpu_ids = physical_sibling_cpu_ids(cpu_ids)
    before = _read_cpu_counters(monitored_cpu_ids)
    time.sleep(0.5)
    after = _read_cpu_counters(monitored_cpu_ids)
    busy_by_cpu = {}
    for cpu_id in monitored_cpu_ids:
        if cpu_id not in before or cpu_id not in after:
            continue
        total_delta = after[cpu_id][0] - before[cpu_id][0]
        idle_delta = after[cpu_id][1] - before[cpu_id][1]
        if total_delta > 0:
            busy_by_cpu[cpu_id] = 100.0 * (1.0 - idle_delta / total_delta)

    busy_percent = list(busy_by_cpu.values())
    busiest = sorted(busy_by_cpu.items(), key=lambda item: (-item[1], item[0]))[:5]

    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_id}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    gpu_memory_mib, gpu_util_percent = (
        float(value.strip()) for value in gpu_line.split(",")
    )
    system_load_1m = os.getloadavg()[0]
    counters_complete = len(busy_by_cpu) == len(monitored_cpu_ids)
    report = {
        "cpu_busy_mean_percent": (
            statistics.fmean(busy_percent) if busy_percent else None
        ),
        "cpu_busy_max_percent": max(busy_percent) if busy_percent else None,
        "cpu_busiest_ids_percent": {str(cpu_id): value for cpu_id, value in busiest},
        "system_load_1m": system_load_1m,
        "selected_cpu_count": len(cpu_ids),
        "monitored_sibling_cpu_count": len(monitored_cpu_ids),
        "monitored_sibling_cpu_ids": monitored_cpu_ids,
        "selected_cpu_counters_complete": counters_complete,
        "system_load_1m_per_cpu": system_load_1m / len(cpu_ids),
        "gpu_memory_used_mib": gpu_memory_mib,
        "gpu_util_percent": gpu_util_percent,
        "clean_required": bool(args.require_clean_resources),
    }
    violations = []
    if not counters_complete:
        violations.append(
            "selected CPU counters are incomplete "
            f"({len(busy_by_cpu)}/{len(monitored_cpu_ids)})"
        )
    if (
        report["cpu_busy_mean_percent"] is not None
        and report["cpu_busy_mean_percent"] > args.max_preflight_cpu_busy_percent
    ):
        violations.append(
            "mean CPU busy "
            f"{report['cpu_busy_mean_percent']:.1f}% > "
            f"{args.max_preflight_cpu_busy_percent:.1f}%"
        )
    if (
        report["cpu_busy_max_percent"] is not None
        and report["cpu_busy_max_percent"] > args.max_preflight_cpu_max_busy_percent
    ):
        violations.append(
            "maximum selected-core CPU busy "
            f"{report['cpu_busy_max_percent']:.1f}% > "
            f"{args.max_preflight_cpu_max_busy_percent:.1f}% "
            f"(busiest={report['cpu_busiest_ids_percent']})"
        )
    if (
        not counters_complete
        and report["system_load_1m_per_cpu"] > args.max_preflight_load_per_cpu
    ):
        violations.append(
            "fallback system load per selected CPU "
            f"{report['system_load_1m_per_cpu']:.3f} > "
            f"{args.max_preflight_load_per_cpu:.3f}"
        )
    if gpu_memory_mib > args.max_preflight_gpu_memory_mib:
        violations.append(
            f"GPU memory {gpu_memory_mib:.0f} MiB > "
            f"{args.max_preflight_gpu_memory_mib:.0f} MiB"
        )
    if gpu_util_percent > args.max_preflight_gpu_util_percent:
        violations.append(
            f"GPU utilization {gpu_util_percent:.1f}% > "
            f"{args.max_preflight_gpu_util_percent:.1f}%"
        )
    report["violations"] = violations
    if violations and args.require_clean_resources:
        raise RuntimeError("Resource preflight failed: " + "; ".join(violations))
    return report


def validate_physical_cpu_ids(
    cpu_ids: list[int], *, allow_smt: bool = False
) -> dict[str, object]:
    output = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    topology: dict[int, tuple[int, int, int]] = {}
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, cpu_socket, node = map(int, line.split(","))
        topology[cpu] = (core, cpu_socket, node)
    missing = [cpu for cpu in cpu_ids if cpu not in topology]
    if missing:
        raise RuntimeError(f"Selected CPUs are unavailable: {missing}")
    physical_ids = {(topology[cpu][1], topology[cpu][0]) for cpu in cpu_ids}
    if not allow_smt and len(physical_ids) != len(cpu_ids):
        raise RuntimeError(
            f"Selected {len(cpu_ids)} logical CPUs but only "
            f"{len(physical_ids)} distinct physical cores"
        )
    numa_node_core_counts = Counter(topology[cpu][2] for cpu in cpu_ids)
    socket_core_counts = Counter(topology[cpu][1] for cpu in cpu_ids)
    return {
        "logical_cpu_ids": cpu_ids,
        "logical_cpu_count": len(cpu_ids),
        "physical_core_count": len(physical_ids),
        "socket_ids": sorted({topology[cpu][1] for cpu in cpu_ids}),
        "numa_node_ids": sorted({topology[cpu][2] for cpu in cpu_ids}),
        "numa_node_core_counts": {
            str(node_id): numa_node_core_counts[node_id]
            for node_id in sorted(numa_node_core_counts)
        },
        "socket_core_counts": {
            str(socket_id): socket_core_counts[socket_id]
            for socket_id in sorted(socket_core_counts)
        },
        "smt_siblings_selected": len(physical_ids) != len(cpu_ids),
    }


def enforce_fixed_64_core_contract(args, cpu_ids: list[int]) -> dict:
    if args.cpu_threads != REQUIRED_PHYSICAL_CPU_CORES:
        raise ValueError("--cpu-threads must be exactly 64")
    if args.faiss_omp_threads != REQUIRED_FAISS_OMP_THREADS:
        raise ValueError("--faiss-omp-threads must be exactly 1")
    if args.faiss_search_workers != REQUIRED_FAISS_SEARCH_WORKERS:
        raise ValueError("--faiss-search-workers must be exactly 64")
    if len(cpu_ids) != REQUIRED_PHYSICAL_CPU_CORES:
        raise RuntimeError(
            "CPU affinity must expose exactly 64 logical CPUs, one per physical core"
        )
    if cpu_ids != list(range(REQUIRED_PHYSICAL_CPU_CORES)):
        raise RuntimeError("CPU affinity must be exactly physical CPUs 0-63")
    topology = validate_physical_cpu_ids(cpu_ids)
    if topology["physical_core_count"] != REQUIRED_PHYSICAL_CPU_CORES:
        raise RuntimeError("CPU affinity must contain 64 physical cores")
    if topology["socket_ids"] != [0]:
        raise RuntimeError("All 64 physical cores must belong to socket 0")
    if topology["numa_node_ids"] != [0, 1, 2, 3]:
        raise RuntimeError("CPU affinity must span exactly NUMA nodes 0-3")
    if topology["numa_node_core_counts"] != {
        "0": 16,
        "1": 16,
        "2": 16,
        "3": 16,
    }:
        raise RuntimeError("Each of NUMA nodes 0-3 must contribute 16 cores")
    if topology["socket_core_counts"] != {"0": 64}:
        raise RuntimeError("Socket 0 must contribute all 64 physical cores")
    return topology


def configure_resources(args: argparse.Namespace) -> list[int]:
    if not args.gpu_id.strip() or "," in args.gpu_id:
        raise ValueError("--gpu-id must select exactly one physical GPU")
    available = sorted(os.sched_getaffinity(0))
    args.cpu_topology = enforce_fixed_64_core_contract(args, available)
    selected = available
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["MNEMORAG_FAISS_OMP_THREADS"] = str(args.faiss_omp_threads)
    os.environ["MNEMORAG_FAISS_SEARCH_WORKERS"] = str(args.faiss_search_workers)
    os.environ["MNEMORAG_FAISS_BATCH_FULL"] = "0"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.faiss_omp_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_PERSISTENT_ASYNC_ENGINE"] = "1"
    os.environ["VLLM_MAX_NUM_SEQS"] = str(max(8, args.microbatch_size))
    if args.max_num_batched_tokens < 0:
        raise ValueError("--max-num-batched-tokens must be nonnegative")
    if args.max_num_batched_tokens:
        os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"] = str(args.max_num_batched_tokens)
    else:
        os.environ.pop("VLLM_MAX_NUM_BATCHED_TOKENS", None)
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    os.environ["VLLM_KV_CACHE_DTYPE"] = args.kv_cache_dtype
    os.environ["MNEMORAG_T_REUSE"] = str(args.tau_reuse)
    os.environ["MNEMORAG_T_GUIDED"] = str(args.tau_guided)
    os.environ["MNEMORAG_ALLOW_THRESHOLD_OVERRIDE"] = "1"
    os.environ["MNEMORAG_FAISS_PREASSIGNED_GUIDANCE"] = (
        "1" if args.preassigned_guidance else "0"
    )
    _, _, async_enabled, component_policy = COMPONENTS[args.components]
    policy = args.kv_policy_override or component_policy
    if args.kv_heat_half_life_s <= 0:
        raise ValueError("--kv-heat-half-life-s must be positive")
    args.effective_kv_policy = policy
    os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = policy
    os.environ["VLLM_KV_HEAT_HALF_LIFE_S"] = str(args.kv_heat_half_life_s)
    os.environ["VLLM_KV_HEAT_SOURCE"] = args.kv_heat_source
    os.environ["VLLM_KV_HEAT_FORMULA"] = args.kv_heat_formula
    os.environ["VLLM_KV_PROFILE_PRIORITY_MODE"] = args.kv_profile_priority_mode
    if args.kv_heat_legacy_time_scale_s <= 0:
        raise ValueError("--kv-heat-legacy-time-scale-s must be positive")
    os.environ["VLLM_KV_HEAT_LEGACY_TIME_SCALE_S"] = str(
        args.kv_heat_legacy_time_scale_s
    )
    # The named formula is authoritative. Do not let an inherited compatibility
    # flag silently switch or conflict with the experiment contract.
    os.environ.pop("VLLM_KV_HEAT_LEGACY_TIMESTAMP_SCORE", None)
    cpu_kv = derive_cpu_kv_layout(args)
    args.cpu_kv_layout = cpu_kv
    for name in (
        "MNEMORAG_CPU_KV_BLOCKS",
        "MNEMORAG_CPU_KV_BLOCK_SIZE",
        "MNEMORAG_CPU_KV_EVICTION_POLICY",
        "VLLM_CPU_KV_BATCH_H2D_TRANSFERS",
        "VLLM_CPU_KV_BATCH_D2H_TRANSFERS",
        "VLLM_CPU_KV_H2D_MAX_GROUP_REQUESTS",
        "VLLM_CPU_KV_H2D_SCHEDULING",
        "VLLM_CPU_KV_SYNC_H2D",
        "VLLM_CPU_KV_DOCUMENT_PREFIX_ONLY",
        "VLLM_CPU_KV_MIN_DOCUMENT_HEAT_TO_STORE",
        "VLLM_CPU_KV_H2D_STREAM_PRIORITY",
        "VLLM_CPU_KV_D2H_STREAM_PRIORITY",
        "VLLM_CPU_KV_HOST_MEMORY_MODE",
        "VLLM_CPU_KV_LAZY_MMAP_DIR",
        "VLLM_CPU_KV_PINNED_TIER_BLOCKS",
        "VLLM_CPU_KV_STAGING_MAX_BLOCKS",
    ):
        os.environ.pop(name, None)
    if cpu_kv["enabled"]:
        os.environ["MNEMORAG_CPU_KV_BLOCKS"] = str(cpu_kv["num_cpu_blocks"])
        os.environ["MNEMORAG_CPU_KV_BLOCK_SIZE"] = str(cpu_kv["block_size_tokens"])
        os.environ["MNEMORAG_CPU_KV_EVICTION_POLICY"] = str(cpu_kv["eviction_policy"])
        os.environ["VLLM_CPU_KV_BATCH_H2D_TRANSFERS"] = (
            "1" if cpu_kv["batch_h2d_transfers"] else "0"
        )
        os.environ["VLLM_CPU_KV_BATCH_D2H_TRANSFERS"] = (
            "1" if cpu_kv["batch_d2h_transfers"] else "0"
        )
        os.environ["VLLM_CPU_KV_H2D_MAX_GROUP_REQUESTS"] = str(
            cpu_kv["h2d_max_group_requests"]
        )
        os.environ["VLLM_CPU_KV_H2D_SCHEDULING"] = str(cpu_kv["h2d_scheduling"])
        os.environ["VLLM_CPU_KV_DOCUMENT_PREFIX_ONLY"] = (
            "1" if cpu_kv["document_prefix_only"] else "0"
        )
        os.environ["VLLM_CPU_KV_MIN_DOCUMENT_HEAT_TO_STORE"] = str(
            cpu_kv["min_document_heat_to_store"]
        )
        os.environ["VLLM_CPU_KV_H2D_STREAM_PRIORITY"] = str(
            cpu_kv["h2d_stream_priority"]
        )
        os.environ["VLLM_CPU_KV_D2H_STREAM_PRIORITY"] = str(
            cpu_kv["d2h_stream_priority"]
        )
        os.environ["VLLM_CPU_KV_HOST_MEMORY_MODE"] = str(cpu_kv["host_memory_mode"])
        if cpu_kv["lazy_mmap_dir"] is not None:
            os.environ["VLLM_CPU_KV_LAZY_MMAP_DIR"] = str(cpu_kv["lazy_mmap_dir"])
        os.environ["VLLM_CPU_KV_PINNED_TIER_BLOCKS"] = str(cpu_kv["pinned_tier_blocks"])
        os.environ["VLLM_CPU_KV_STAGING_MAX_BLOCKS"] = str(
            cpu_kv["staging_buffer_blocks"]
        )
        # The connector implements async/sync/adaptive scheduling per step.
        # Disable the legacy handler-wide synchronization override so it
        # cannot silently turn adaptive or async runs into global sync runs.
        os.environ["VLLM_CPU_KV_SYNC_H2D"] = "0"
        print(
            "[MnemoRAG] CPU KV offload enabled: "
            f"capacity={cpu_kv['capacity_gb']:.3f} GB "
            f"({cpu_kv['capacity_gib']:.3f} GiB), "
            f"block_size={cpu_kv['block_size_tokens']} tokens, "
            f"block_bytes={cpu_kv['block_bytes']}, "
            f"kv_cache_dtype={cpu_kv['kv_cache_dtype']}, "
            f"num_cpu_blocks={cpu_kv['num_cpu_blocks']}, "
            f"eviction_policy={cpu_kv['eviction_policy']}, "
            f"batch_h2d={cpu_kv['batch_h2d_transfers']}, "
            f"batch_d2h={cpu_kv['batch_d2h_transfers']}, "
            f"h2d_max_group_requests={cpu_kv['h2d_max_group_requests']}, "
            f"h2d_scheduling={cpu_kv['h2d_scheduling']}, "
            f"document_prefix_only={cpu_kv['document_prefix_only']}, "
            "min_document_heat_to_store="
            f"{cpu_kv['min_document_heat_to_store']}, "
            f"h2d_stream_priority={cpu_kv['h2d_stream_priority']}, "
            f"d2h_stream_priority={cpu_kv['d2h_stream_priority']}",
            f"host_memory_mode={cpu_kv['host_memory_mode']} "
            f"lazy_allocation={cpu_kv['lazy_allocation']} "
            f"lazy_mmap_dir={cpu_kv['lazy_mmap_dir']} "
            "lazy_backing_required_bytes="
            f"{cpu_kv['lazy_backing_required_bytes']} "
            f"pinned_tier_gb={cpu_kv['pinned_tier_gb']:.3f} "
            f"pinned_tier_blocks={cpu_kv['pinned_tier_blocks']} "
            f"staging_buffer_blocks={cpu_kv['staging_buffer_blocks']}",
            flush=True,
        )
    args.resource_preflight = resource_preflight(args, selected)
    args.numa_policy = current_numa_policy()
    args.numa_pages_before = current_numa_page_counts()
    print(
        "[MnemoRAG] Resource preflight: "
        + json.dumps(args.resource_preflight, sort_keys=True),
        flush=True,
    )
    print(
        "[MnemoRAG] NUMA policy: " + json.dumps(args.numa_policy, sort_keys=True),
        flush=True,
    )
    return selected


def load_sample(
    path: Path,
    count: int,
    seed: int,
    *,
    explicit_indices=None,
) -> tuple[list[dict], list[int], int]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if explicit_indices is None:
        if count > len(rows):
            raise ValueError(f"Requested {count} rows from {len(rows)} available rows")
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(rows)), count))
    else:
        indices = [int(index) for index in explicit_indices]
        if len(indices) != count:
            raise ValueError(
                f"Explicit sample has {len(indices)} rows; expected {count}"
            )
        if any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError("Explicit sample index is outside the dataset")
    return [rows[index] for index in indices], indices, len(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for row in rows:
            payload = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            digest.update(payload)
            stream.write(payload)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "count": len(rows),
        "sha256": digest.hexdigest(),
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def online_prefix_cache_windows(
    records: list[dict],
    *,
    window_seconds: float = 10.0,
) -> dict:
    """Summarize causal per-request prefix hits in fixed scheduler windows."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    usable = [
        record
        for record in records
        if record.get("first_scheduled_s") is not None
        and record.get("prompt_tokens") is not None
        and record.get("cached_tokens") is not None
    ]
    if not usable:
        return {
            "window_seconds": float(window_seconds),
            "timestamp_basis": "first_scheduled_s",
            "window_count": 0,
            "request_count": 0,
            "mean_window_hit_rate": None,
            "max_window_hit_rate": None,
            "overall_token_weighted_hit_rate": None,
            "samples": [],
        }

    origin = min(float(record["first_scheduled_s"]) for record in usable)
    windows: dict[int, dict[str, int]] = {}
    for record in usable:
        offset = max(0.0, float(record["first_scheduled_s"]) - origin)
        window_index = int(math.floor(offset / window_seconds))
        accumulator = windows.setdefault(
            window_index,
            {"request_count": 0, "prompt_tokens": 0, "cached_tokens": 0},
        )
        accumulator["request_count"] += 1
        accumulator["prompt_tokens"] += int(record["prompt_tokens"])
        accumulator["cached_tokens"] += min(
            int(record["prompt_tokens"]), int(record["cached_tokens"])
        )

    samples = []
    for window_index in sorted(windows):
        accumulator = windows[window_index]
        prompt_tokens = accumulator["prompt_tokens"]
        hit_rate = (
            accumulator["cached_tokens"] / prompt_tokens if prompt_tokens else 0.0
        )
        samples.append(
            {
                "window_index": window_index,
                "start_offset_s": float(window_index * window_seconds),
                "end_offset_s": float((window_index + 1) * window_seconds),
                **accumulator,
                "hit_rate": float(hit_rate),
            }
        )

    total_prompt_tokens = sum(sample["prompt_tokens"] for sample in samples)
    total_cached_tokens = sum(sample["cached_tokens"] for sample in samples)
    return {
        "window_seconds": float(window_seconds),
        "timestamp_basis": "first_scheduled_s",
        "window_count": len(samples),
        "request_count": sum(sample["request_count"] for sample in samples),
        "mean_window_hit_rate": statistics.fmean(
            sample["hit_rate"] for sample in samples
        ),
        "max_window_hit_rate": max(sample["hit_rate"] for sample in samples),
        "overall_token_weighted_hit_rate": (
            total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
        ),
        "samples": samples,
    }


def online_prefix_cache_recent_request_window(
    records: list[dict],
    *,
    requested_window_size: int = 1000,
) -> dict:
    """Compute vLLM-style token hits over the most recent requests available."""
    if requested_window_size <= 0:
        raise ValueError("requested_window_size must be positive")
    usable = sorted(
        (
            record
            for record in records
            if record.get("first_scheduled_s") is not None
            and record.get("prompt_tokens") is not None
            and record.get("cached_tokens") is not None
        ),
        key=lambda record: float(record["first_scheduled_s"]),
    )
    observed = usable[-requested_window_size:]
    prompt_tokens = sum(int(record["prompt_tokens"]) for record in observed)
    cached_tokens = sum(
        min(int(record["prompt_tokens"]), int(record["cached_tokens"]))
        for record in observed
    )
    return {
        "requested_window_size": int(requested_window_size),
        "observed_window_size": len(observed),
        "full_window_available": len(observed) == requested_window_size,
        "timestamp_basis": "first_scheduled_s",
        "prompt_tokens": int(prompt_tokens),
        "cached_tokens": int(cached_tokens),
        "hit_rate": (float(cached_tokens / prompt_tokens) if prompt_tokens else None),
    }


_TRANSFER_CUMULATIVE_FIELDS = {
    "limited_requests",
    "tail_blocks_skipped",
    "cold_admission_skipped_requests",
    "cold_admission_blocks_skipped",
    "document_heat_observed_requests",
    "document_heat_above_one_requests",
    "scheduled_request_operations",
    "scheduled_spec_units",
    "worker_completed_jobs",
    "worker_completed_requests",
    "worker_transferred_blocks",
    "worker_observed_total_s",
    "worker_submit_total_s",
    "cuda_total_s",
    "scheduler_wait_completed_requests",
    "scheduler_wait_total_s",
    "async_jobs",
    "sync_jobs",
    "deferred_sync_jobs",
    "scheduling_requests",
    "metadata_delivery_total_s",
    "defer_delay_total_s",
    "blocking_wait_total_s",
}


def transfer_stats_delta(before: dict | None, after: dict | None) -> dict | None:
    """Return measurement-only CPU-KV counters while preserving cache state."""
    if after is None:
        return None
    before = before or {}
    result = copy.deepcopy(after)
    for section_name in ("d2h_admission", "h2d_reload", "d2h_offload"):
        section = result.get(section_name)
        if not isinstance(section, dict):
            continue
        previous = before.get(section_name) or {}
        for field in _TRANSFER_CUMULATIVE_FIELDS:
            value = section.get(field)
            old_value = previous.get(field, 0)
            if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
                section[field] = max(0, value - old_value)
        completed = int(section.get("worker_completed_jobs", 0) or 0)
        if section_name in {"h2d_reload", "d2h_offload"}:
            for total_name, mean_name in (
                ("worker_observed_total_s", "worker_observed_mean_s"),
                ("worker_submit_total_s", "worker_submit_mean_s"),
                ("cuda_total_s", "cuda_mean_s"),
            ):
                section[mean_name] = (
                    float(section.get(total_name, 0.0)) / completed
                    if completed
                    else 0.0
                )
        if section_name == "h2d_reload":
            waited = int(section.get("scheduler_wait_completed_requests", 0) or 0)
            section["scheduler_wait_mean_s"] = (
                float(section.get("scheduler_wait_total_s", 0.0)) / waited
                if waited
                else 0.0
            )
    result["counter_scope"] = "measurement_only_after_online_warmup"
    result["maximum_latency_scope"] = "warmup_and_measurement"
    return result


def retrieval_profile_measurement_delta(
    before: dict | None, after: dict | None
) -> dict | None:
    """Return similarity-aware counters for post-warmup requests only."""
    if after is None:
        return None
    before = before or {}
    result = {
        "counter_scope": "measurement_only_after_online_warmup",
        "batches": max(
            0,
            int(after.get("batches", 0) or 0) - int(before.get("batches", 0) or 0),
        ),
        "action_counts": {},
    }
    previous_actions = before.get("action_counts") or {}
    current_actions = after.get("action_counts") or {}
    for action in ("exact", "full", "guided", "reuse"):
        result["action_counts"][action] = max(
            0,
            int(current_actions.get(action, 0) or 0)
            - int(previous_actions.get(action, 0) or 0),
        )
    result["sar_decision_actions"] = sum(
        result["action_counts"][action] for action in ("full", "guided", "reuse")
    )
    result["request_actions"] = sum(result["action_counts"].values())
    return result


def array_sha256(value) -> str:
    numpy = __import__("numpy")
    contiguous = numpy.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(numpy.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_retrieval_id_sets(rows: list[object]) -> list[object]:
    """Preserve request/round boundaries while removing document-order effects."""

    def normalize(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            normalized = [normalize(item) for item in value]
            if all(not isinstance(item, (list, dict)) for item in normalized):
                return sorted(
                    normalized, key=lambda item: (type(item).__name__, str(item))
                )
            return normalized
        if hasattr(value, "item"):
            return value.item()
        return value

    return [normalize(row) for row in rows]


def load_retrieval_reference(
    path: Path,
    expected_requests: int,
    topk: int,
) -> tuple[list[list[int]], list[list[float]], str]:
    reference_ids: list[list[int]] = []
    reference_scores: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ids = [int(value) for value in json.loads(row["retrieval_idxs"])]
            scores = [float(value) for value in json.loads(row["retrieval_scores"])]
            if len(ids) != topk or len(scores) != topk:
                raise ValueError(
                    f"Retrieval reference row has {len(ids)} IDs and "
                    f"{len(scores)} scores; expected top-k={topk}"
                )
            reference_ids.append(ids)
            reference_scores.append(scores)
    if len(reference_ids) != expected_requests:
        raise ValueError(
            f"Retrieval reference contains {len(reference_ids)} requests; "
            f"expected {expected_requests}"
        )
    return reference_ids, reference_scores, canonical_sha256(reference_ids)


def load_retrieval_replay_csv(
    path: Path,
    *,
    expected_count: int,
    expected_topk: int,
) -> list[dict[str, object]]:
    """Load the legacy query-checked replay format used by paired runs."""
    replay = []
    with path.open(newline="", encoding="utf-8") as stream:
        for expected_index, row in enumerate(csv.DictReader(stream)):
            request_index = int(row["request_index"])
            if request_index != expected_index:
                raise ValueError(
                    "Retrieval replay request indices must be contiguous and "
                    f"ordered; expected {expected_index}, observed {request_index}"
                )
            ids = [int(value) for value in json.loads(row["retrieval_idxs"])]
            scores = [float(value) for value in json.loads(row["retrieval_scores"])]
            if len(ids) != expected_topk or len(scores) != expected_topk:
                raise ValueError(
                    f"Retrieval replay row has {len(ids)} IDs and "
                    f"{len(scores)} scores; expected top-k={expected_topk}"
                )
            replay.append(
                {
                    "request_index": request_index,
                    "question": str(row["question"]),
                    "ids": ids,
                    "scores": scores,
                }
            )
    if len(replay) != expected_count:
        raise ValueError(
            f"Retrieval replay contains {len(replay)} requests; "
            f"expected {expected_count}"
        )
    return replay


def install_retrieval_replay(retriever, replay, load_documents):
    """Install deterministic replay while rejecting query-order drift."""
    state = {"cursor": 0}

    def replay_batch_search(queries, *positional, **keywords):
        if positional:
            raise TypeError("retrieval replay requires keyword search options")
        query_list = [queries] if isinstance(queries, str) else list(queries)
        begin = state["cursor"]
        end = begin + len(query_list)
        if end > len(replay):
            raise RuntimeError("retrieval replay exhausted before the workload")
        rows = replay[begin:end]
        for offset, (query, row) in enumerate(zip(query_list, rows)):
            if str(query) != row["question"]:
                raise RuntimeError(
                    "retrieval replay query mismatch at request "
                    f"{begin + offset}: expected {row['question']!r}, "
                    f"observed {str(query)!r}"
                )
        ids = [list(row["ids"]) for row in rows]
        scores = [list(row["scores"]) for row in rows]
        flat_ids = [document_id for request_ids in ids for document_id in request_ids]
        flat_documents = load_documents(retriever.corpus, flat_ids)
        topk = int(getattr(retriever, "topk", len(ids[0]) if ids else 0))
        documents = [
            flat_documents[offset : offset + topk]
            for offset in range(0, len(flat_documents), topk)
        ]
        state["cursor"] = end
        retriever.last_request_actions = ["replay"] * len(query_list)
        return_score = bool(keywords.get("return_score", False))
        return_idx = bool(keywords.get("return_idx", False))
        if return_score and return_idx:
            return documents, scores, ids
        if return_score:
            return documents, scores
        if return_idx:
            return documents, ids
        return documents

    retriever.batch_search = replay_batch_search
    return state


def source_order_prompt_fingerprints(dataset, tokenizer) -> dict[str, object]:
    """Fingerprint exact prompts and token IDs in original request order."""

    prompts: list[str] = []
    for request_index, item in enumerate(dataset):
        prompt = item.output.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError(f"Missing generation prompt for request {request_index}")
        prompts.append(prompt)

    prompt_token_hashes: list[str] = []
    prompt_token_lengths: list[int] = []
    batch_size = 256
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        for token_ids in input_ids:
            token_ids = list(token_ids)
            prompt_token_hashes.append(canonical_sha256(token_ids))
            prompt_token_lengths.append(len(token_ids))

    return {
        "request_count": len(prompts),
        "prompt_text_sha256": canonical_sha256(prompts),
        "prompt_token_ids_sha256": canonical_sha256(prompt_token_hashes),
        "prompt_lengths_sha256": canonical_sha256(prompt_token_lengths),
    }


def current_numa_policy() -> dict[str, str]:
    output = subprocess.check_output(
        ["numactl", "--show"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    parsed = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().replace(" ", "_")] = value.strip()
    return {"raw": output.strip(), **parsed}


def current_numa_page_counts() -> dict[str, object]:
    pages = Counter()
    policies = Counter()
    with open("/proc/self/numa_maps", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) > 1:
                policies[fields[1]] += 1
            for node, count in re.findall(r"\bN(\d+)=(\d+)\b", line):
                pages[int(node)] += int(count)
    total = sum(pages.values())
    local = sum(pages[node] for node in (0, 1, 2, 3))
    remote = sum(pages[node] for node in (4, 5, 6, 7))
    return {
        "pages_by_node": {str(node): pages[node] for node in sorted(pages)},
        "total_pages": total,
        "socket0_pages": local,
        "socket1_pages": remote,
        "socket1_page_fraction": remote / total if total else 0.0,
        "policies": dict(sorted(policies.items())),
    }


def collect_quality_metric_values(dataset, expected_count: int):
    """Collect one finite Recall/F1/ACC observation for every request."""
    names = ("retrieval_recall", "f1", "acc")
    values = {name: [] for name in names}
    observed = 0
    for request_index, item in enumerate(dataset):
        metrics = item.output.get("metric_score") or {}
        missing = [name for name in names if metrics.get(name) is None]
        if missing:
            raise RuntimeError(
                "Missing quality metrics for request "
                f"{request_index}: {', '.join(missing)}"
            )
        for name in names:
            value = float(metrics[name])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {name} for request {request_index}: {value}"
                )
            values[name].append(value)
        observed += 1
    if observed != int(expected_count):
        raise RuntimeError(
            "Quality metric denominator mismatch: "
            f"expected {expected_count} requests, observed {observed}"
        )
    return values


def write_request_rows(path: Path, dataset) -> None:
    fieldnames = (
        "request_index",
        "id",
        "question",
        "pred",
        "retrieval_idxs",
        "retrieval_scores",
        "f1",
        "acc",
        "retrieval_recall_at_k",
        "retrieval_k",
        "scheduled_arrival_offset_s",
        "arrival_offset_s",
        "actual_arrival_offset_s",
        "arrival_schedule_lag_s",
        "finish_offset_s",
        "e2e_latency_s",
        "post_admission_latency_s",
        "scheduler_admission_wait_s",
        "scheduler_queue_wait_s",
        "irg_iterations",
        "irg_stop_reason",
        "recomp_retrieval_batch_time_s",
        "recomp_compression_batch_time_s",
        "recomp_answer_batch_time_s",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for request_index, item in enumerate(dataset):
            output = item.output
            metrics = output.get("metric_score") or {}
            writer.writerow(
                {
                    "request_index": request_index,
                    "id": item.data.get("id", request_index),
                    "question": item.data.get("question", ""),
                    "pred": output.get("pred", ""),
                    "retrieval_idxs": json.dumps(
                        output.get("retrieval_idxs"), separators=(",", ":")
                    ),
                    "retrieval_scores": json.dumps(
                        output.get("retrieval_score"), separators=(",", ":")
                    ),
                    "f1": metrics.get("f1"),
                    "acc": metrics.get("acc"),
                    "retrieval_recall_at_k": metrics.get("retrieval_recall"),
                    "retrieval_k": len(output.get("retrieval_result") or ()),
                    "scheduled_arrival_offset_s": output.get(
                        "scheduled_arrival_offset_s"
                    ),
                    "arrival_offset_s": output.get("arrival_offset_s"),
                    "actual_arrival_offset_s": output.get("actual_arrival_offset_s"),
                    "arrival_schedule_lag_s": output.get("arrival_schedule_lag_s"),
                    "finish_offset_s": output.get("finish_offset_s"),
                    "e2e_latency_s": output.get("e2e_latency_s"),
                    "post_admission_latency_s": output.get("post_admission_latency_s"),
                    "scheduler_admission_wait_s": output.get(
                        "scheduler_admission_wait_s"
                    ),
                    "scheduler_queue_wait_s": output.get("scheduler_queue_wait_s"),
                    "irg_iterations": output.get("irg_iterations"),
                    "irg_stop_reason": output.get("irg_stop_reason"),
                    "recomp_retrieval_batch_time_s": output.get(
                        "recomp_retrieval_batch_time_s"
                    ),
                    "recomp_compression_batch_time_s": output.get(
                        "recomp_compression_batch_time_s"
                    ),
                    "recomp_answer_batch_time_s": output.get(
                        "recomp_answer_batch_time_s"
                    ),
                }
            )
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("vllm", "torch", "faiss", "transformers"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - diagnostic only
            versions[package] = f"unavailable: {error}"
    try:
        versions["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        versions["git_commit"] = "unavailable"
    return versions


async def run(args: argparse.Namespace, selected_cpus: list[int]) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "examples/quick_start"))
    # The repository contains partial vLLM patch sources, so relying on cwd
    # can resolve ``vllm`` as an empty namespace package.  Pin the complete
    # Resolve the complete MnemoRAG vLLM runtime before loading the large index.
    if not (MNEMORAG_VLLM_ROOT / "vllm/__init__.py").is_file():
        raise RuntimeError(f"MnemoRAG vLLM runtime is incomplete: {MNEMORAG_VLLM_ROOT}")
    sys.path.insert(0, str(MNEMORAG_VLLM_ROOT))
    import vllm

    vllm_file = Path(vllm.__file__ or "").resolve()
    if MNEMORAG_VLLM_ROOT.resolve() not in vllm_file.parents:
        raise RuntimeError(
            f"Resolved vLLM outside the locked MnemoRAG runtime: {vllm_file}"
        )
    from utils import poisson_persistent_workflow_run

    from mnemorag.config import Config
    from mnemorag.dataset.dataset import Dataset
    from mnemorag.pipeline import AsyncPipeline
    from mnemorag.pipeline.utils import DocumentPool
    from mnemorag.prompt import PromptTemplate
    from mnemorag.retriever.utils import load_docs
    from mnemorag.utils import get_retriever

    similarity, hotness, async_enabled, component_policy = COMPONENTS[args.components]
    similarity_nprobe_min, similarity_nprobe_max = resolve_similarity_nprobe_bounds(
        args.nprobe,
        args.similarity_nprobe_min_fraction,
        args.similarity_nprobe_max_fraction,
    )
    if args.similarity_nprobe_gamma is not None and args.similarity_nprobe_gamma <= 0:
        raise ValueError("similarity nprobe gamma must be positive")
    policy = (
        getattr(args, "effective_kv_policy", None)
        or args.kv_policy_override
        or component_policy
    )
    if args.order_mode == "original":

        def identity_reorder(self, doc_idxs, doc_contents):
            return doc_contents, doc_idxs

        DocumentPool.reorder_by_hotness = identity_reorder
    if args.request_cycle_size < 0:
        raise ValueError("request_cycle_size must be non-negative")
    if args.arrival_trace is not None and args.request_cycle_size:
        raise ValueError("--arrival-trace cannot be combined with --request-cycle-size")
    arrival_spec = ArrivalTraceSpec(
        request_count=args.num_requests,
        request_rate=args.arrival_rate,
        burstiness=args.arrival_burstiness,
        random_seed=args.seed,
        normalize_duration=args.normalize_arrival_trace,
    )
    arrival_offsets = None
    arrival_manifest = None
    trace_sample_indices = None
    if args.arrival_trace is not None:
        arrival_offsets, arrival_manifest = load_trace_bundle(args.arrival_trace)
        trace_sample_indices = load_trace_sample_indices(args.arrival_trace)
    sample_count = (
        min(args.num_requests, args.request_cycle_size)
        if args.request_cycle_size
        else args.num_requests
    )
    rows, sample_indices, source_count = load_sample(
        DATASET_PATHS[args.dataset],
        sample_count,
        args.seed,
        explicit_indices=trace_sample_indices,
    )
    warmup_rows = None
    warmup_sample_indices: list[int] = []
    if args.online_warmup_indices is not None:
        raw_warmup_indices = __import__("numpy").asarray(
            __import__("numpy").load(args.online_warmup_indices, allow_pickle=False),
            dtype="<i8",
        )
        if raw_warmup_indices.ndim != 1 or not len(raw_warmup_indices):
            raise ValueError("online warmup indices must be a non-empty vector")
        warmup_rows, warmup_sample_indices, warmup_source_count = load_sample(
            DATASET_PATHS[args.dataset],
            len(raw_warmup_indices),
            args.seed,
            explicit_indices=raw_warmup_indices,
        )
        if warmup_source_count != source_count:
            raise ValueError("warmup and measurement source counts differ")
        if len(set(warmup_sample_indices)) != len(warmup_sample_indices):
            raise ValueError("online warmup indices must be unique")
        overlap = set(warmup_sample_indices).intersection(sample_indices)
        if overlap:
            raise ValueError(f"online warmup overlaps {len(overlap)} measured rows")
    if args.request_cycle_size:
        rows = [
            copy.deepcopy(rows[index % len(rows)]) for index in range(args.num_requests)
        ]
        sample_indices = [
            sample_indices[index % len(sample_indices)]
            for index in range(args.num_requests)
        ]
    if args.arrival_trace is not None:
        validate_trace_workload(
            arrival_manifest,
            arrival_spec,
            sample_indices=sample_indices,
            dataset_sha256=file_sha256(DATASET_PATHS[args.dataset]),
        )
        arrival_trace_source = str(args.arrival_trace.resolve())
    else:
        arrival_offsets = build_arrival_offsets(arrival_spec)
        arrival_manifest = build_trace_manifest(
            arrival_spec,
            arrival_offsets,
            sample_indices=sample_indices,
            dataset_sha256=file_sha256(DATASET_PATHS[args.dataset]),
        )
        arrival_trace_source = "generated"
    retrieval_reference_ids = None
    retrieval_reference_scores = None
    retrieval_reference_sha256 = None
    if args.retrieval_reference_csv is not None:
        (
            retrieval_reference_ids,
            retrieval_reference_scores,
            retrieval_reference_sha256,
        ) = load_retrieval_reference(
            args.retrieval_reference_csv,
            args.num_requests,
            args.topk,
        )
    config = Config(
        config_dict={
            "split": ["test"],
            "dataset_name": args.dataset,
            "data_dir": str(ASSET_ROOT / "qa/datasets"),
            "index_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
            ),
            "corpus_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
            ),
            "model2path": {
                "e5": str(ASSET_ROOT / "models/e5-large-v2"),
                "llama3-8B-instruct": str(ASSET_ROOT / "models/llama3-8B-instruct"),
            },
            "generator_model": "llama3-8B-instruct",
            "retrieval_method": "e5",
            "framework": "vllm",
            "metrics": ["retrieval_recall", "f1", "acc"],
            "metric_setting": {
                "retrieval_recall_topk": args.topk,
            },
            "retrieval_topk": args.topk,
            "retrieval_batch_size": args.retrieval_batch_size,
            "generator_batch_size": 256,
            "generator_max_input_len": args.max_model_len,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "ignore_eos": args.ignore_eos,
            },
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "enable_prefix_caching": args.prefix_caching,
            "ivf_nprobe": args.nprobe,
            "enable_similarity_aware": similarity,
            "enable_preassigned_guidance": args.preassigned_guidance,
            "enable_hotness_sort": hotness,
            "enable_document_kv_heat": args.document_kv_heat,
            "similarity_t_reuse": args.tau_reuse,
            "similarity_t_guided": args.tau_guided,
            "similarity_nprobe_min": similarity_nprobe_min,
            "similarity_nprobe_max": similarity_nprobe_max,
            "similarity_nprobe_gamma": args.similarity_nprobe_gamma,
            "similarity_nprobe_curve": args.similarity_nprobe_curve,
            "query_base_exact_lookup": args.query_base_exact_lookup,
            "gpu_id": args.gpu_id,
            "gpu_num": 1,
            "cpu_threads": args.cpu_threads,
            "faiss_omp_threads": args.faiss_omp_threads,
            "faiss_search_workers": args.faiss_search_workers,
            "enable_single_query_probe_parallelism": (args.system_label == "mnemorag"),
            "faiss_mmap_index": args.faiss_mmap_index,
            "async_retrieval_policy": args.async_retrieval_policy,
            "async_cross_microbatch_pipeline": args.cross_microbatch_pipeline,
            "direct_vllm_dispatch": args.direct_vllm_dispatch,
            "cache_aware_prefill_batching": args.cache_aware_prefill_batching,
            "async_fixed_delay_ms": args.async_fixed_delay_ms,
            "async_adaptive_ema_alpha": args.async_adaptive_ema_alpha,
            "async_adaptive_max_prefill_overlap_fraction": (
                args.async_adaptive_max_prefill_overlap_fraction
            ),
            "async_adaptive_safety_margin_ms": (args.async_adaptive_safety_margin_ms),
            "irg_max_iter": args.irg_rounds,
            "recomp_compression_max_tokens": (DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS),
            "recomp_answer_max_tokens": args.max_tokens,
            "seed": args.seed,
            "disable_save": True,
        }
    )
    prompt_template = build_prompt_template(config, args.order_mode, PromptTemplate)
    if args.drop_loaded_input_page_cache and args.faiss_mmap_index:
        raise ValueError(
            "--drop-loaded-input-page-cache requires an eagerly loaded index"
        )

    page_cache_release = None
    preloaded_retriever = None
    retriever_preload_s = None
    startup_reservation = getattr(args, "startup_gpu_reservation_handle", None)
    if startup_reservation is not None and not args.drop_loaded_input_page_cache:
        raise ValueError(
            "--startup-gpu-reservation-mib requires "
            "--drop-loaded-input-page-cache so the retriever is preloaded"
        )
    if args.drop_loaded_input_page_cache and (
        args.cpu_kv_layout["enabled"] or startup_reservation is not None
    ):
        preload_started = time.perf_counter()
        preloaded_retriever = get_retriever(config)
        retriever_preload_s = time.perf_counter() - preload_started
        if startup_reservation is not None:
            import torch

            handoff_mib = torch.cuda.memory_reserved() / 1024**2
            startup_reservation["report"]["handoff_parent_reserved_mib"] = handoff_mib
            if handoff_mib <= MIN_GPU_RESERVATION_HANDOFF_MIB:
                raise RuntimeError(
                    "Retriever did not acquire enough GPU memory before "
                    "startup reservation release: "
                    f"{handoff_mib:.3f} MiB"
                )
            args.startup_gpu_reservation_report = release_startup_gpu_reservation(
                startup_reservation
            )
        page_cache_release = release_loaded_input_page_cache(
            [Path(config["index_path"]), Path(config["corpus_path"])]
        )
        print(
            "[MnemoRAG] Retriever-first CPU KV initialization: "
            f"retriever_preload_s={retriever_preload_s:.6f}, "
            "input page cache released before generator startup",
            flush=True,
        )
        print(
            "[MnemoRAG] Input page-cache release: "
            + json.dumps(page_cache_release, sort_keys=True),
            flush=True,
        )
        if page_cache_release["status"] != "complete":
            raise RuntimeError(
                f"Unable to release loaded input page cache: {page_cache_release}"
            )

    pipeline_init_started = time.perf_counter()
    pipeline = AsyncPipeline(
        config,
        prompt_template=prompt_template,
        retriever=preloaded_retriever,
    )
    pipeline_init_s = time.perf_counter() - pipeline_init_started
    if pipeline.generator.tensor_parallel_size != 1:
        raise RuntimeError("The evaluation must use exactly one GPU")
    effective_instruction = pipeline.retriever.instruction
    if effective_instruction is None:
        from mnemorag.retriever.utils import set_default_instruction

        effective_instruction = set_default_instruction(
            pipeline.retriever.retrieval_method, is_query=True
        )
    else:
        effective_instruction = str(effective_instruction).strip() + " "
    encoder_contract = {
        "model": Path(pipeline.retriever.retrieval_model_path).name,
        "pooling": pipeline.retriever.pooling_method,
        "precision": ("fp16" if pipeline.retriever.use_fp16 else "fp32"),
        "max_length": int(pipeline.retriever.query_max_length),
        "query_instruction": effective_instruction,
        "l2_normalization": True,
        "output_dtype": "float32",
    }
    expected_encoder_contract = {
        "model": "e5-large-v2",
        "pooling": "mean",
        "precision": "fp16",
        "max_length": 128,
        "query_instruction": "query: ",
        "l2_normalization": True,
        "output_dtype": "float32",
    }
    if encoder_contract != expected_encoder_contract:
        raise RuntimeError(
            f"Canonical E5 encoder contract mismatch: {encoder_contract}"
        )
    expected_vllm_contract = {
        "max_num_seqs": args.microbatch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_prefix_caching": bool(args.prefix_caching),
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    if args.max_num_batched_tokens > 0:
        expected_vllm_contract["max_num_batched_tokens"] = args.max_num_batched_tokens
    vllm_engine_contract = validate_effective_vllm_contract(
        pipeline.generator.model,
        expected_vllm_contract,
    )
    if args.drop_loaded_input_page_cache and page_cache_release is None:
        page_cache_release = release_loaded_input_page_cache(
            [Path(config["index_path"]), Path(config["corpus_path"])]
        )
        print(
            "[MnemoRAG] Input page-cache release: "
            + json.dumps(page_cache_release, sort_keys=True),
            flush=True,
        )
        if page_cache_release["status"] != "complete":
            raise RuntimeError(
                f"Unable to release loaded input page cache: {page_cache_release}"
            )

    online_warmup = None
    warmup_kv_cache_stats = None
    warmup_retrieval_profile = None
    if warmup_rows is not None:
        warmup_monitor = getattr(args, "runtime_resource_monitor", None)
        if warmup_monitor is not None:
            warmup_monitor.checkpoint("online_warmup:start")
        warmup_started = time.perf_counter()
        warmup_output = await poisson_persistent_workflow_run(
            Dataset(data=warmup_rows),
            pipeline,
            args.arrival_rate,
            workflow=args.workflow,
            do_eval=False,
            worker_count=args.worker_count if async_enabled else 1,
            microbatch_max_size=args.microbatch_size,
            arrival_burstiness=args.arrival_burstiness,
            normalize_arrival_trace=args.normalize_arrival_trace,
            random_seed=args.seed ^ 0x5A17,
            record_request_timing=False,
        )
        warmup_elapsed_s = time.perf_counter() - warmup_started
        if len(warmup_output) != len(warmup_rows):
            raise RuntimeError("online warmup did not complete every request")
        warmup_kv_cache_stats = await pipeline.generator.get_kv_cache_stats()
        warmup_retrieval_profile = (
            pipeline.retriever.retrieval_profile()
            if hasattr(pipeline.retriever, "retrieval_profile")
            else None
        )
        pipeline.generator.exact_prefix_cached_tokens = 0
        pipeline.generator.exact_prefix_prompt_tokens = 0
        pipeline.last_workflow_scheduler_stats = {}
        pipeline.last_async_scheduler_stats = None
        pipeline.last_async_scheduler_trace = None
        online_warmup = {
            "status": "complete",
            "request_count": len(warmup_rows),
            "elapsed_s": warmup_elapsed_s,
            "arrival_rate_rps": float(args.arrival_rate),
            "workflow": str(args.workflow),
            "indices_path": str(args.online_warmup_indices.resolve()),
            "indices_sha256": file_sha256(args.online_warmup_indices),
            "sample_indices_array_sha256": canonical_sha256(
                list(map(int, warmup_sample_indices))
            ),
            "measurement_excludes_warmup": True,
            "same_engine_instance_as_measurement": True,
            "cache_state_preserved": True,
        }
        if warmup_monitor is not None:
            warmup_monitor.checkpoint("online_warmup:complete")

    retrieval_latencies_s: list[float] = []
    retrieval_batch_records: list[dict[str, object]] = []
    generation_stage_records: list[dict[str, float]] = []
    generation_batch_records: list[dict[str, float]] = []
    prefix_request_hit_rates: list[float] = []
    prefix_request_cached_tokens: list[int] = []
    prefix_request_prompt_tokens: list[int] = []
    prefix_periodic_recent_samples: list[dict[str, float | int]] = []
    prefix_recent_requests: deque[tuple[int, int]] = deque()
    prefix_recent_cached_tokens = 0
    prefix_recent_prompt_tokens = 0
    prefix_periodic_origin_s: float | None = None
    prefix_next_periodic_sample_s = 1.0
    retrieval_reference_cursor = 0
    if args.record_stage_breakdown or retrieval_reference_ids is not None:
        original_batch_search = pipeline.retriever.batch_search

        def instrumented_batch_search(queries, *positional, **keywords):
            nonlocal retrieval_reference_cursor
            stage_started = time.perf_counter()
            result = original_batch_search(queries, *positional, **keywords)
            if retrieval_reference_ids is not None:
                reference_end = retrieval_reference_cursor + len(queries)
                if reference_end > len(retrieval_reference_ids):
                    raise RuntimeError(
                        "Retrieval reference exhausted before the workload"
                    )
                batch_ids = retrieval_reference_ids[
                    retrieval_reference_cursor:reference_end
                ]
                batch_scores = retrieval_reference_scores[
                    retrieval_reference_cursor:reference_end
                ]
                flat_ids = [
                    document_id
                    for request_ids in batch_ids
                    for document_id in request_ids
                ]
                flat_results = load_docs(pipeline.retriever.corpus, flat_ids)
                batch_results = [
                    flat_results[offset : offset + args.topk]
                    for offset in range(0, len(flat_results), args.topk)
                ]
                result = (batch_results, batch_scores, batch_ids)
                retrieval_reference_cursor = reference_end
            stage_elapsed = time.perf_counter() - stage_started
            if args.record_stage_breakdown:
                retrieval_latencies_s.extend([stage_elapsed] * len(queries))
                retrieval_batch_records.append(
                    {
                        "batch_index": len(retrieval_batch_records),
                        "request_count": len(queries),
                        "total_s": stage_elapsed,
                        "phase_s": dict(
                            getattr(
                                pipeline.retriever,
                                "last_retrieval_phase_times",
                                {},
                            )
                        ),
                    }
                )
            return result

        pipeline.retriever.batch_search = instrumented_batch_search

    if args.record_stage_breakdown:
        stage_format_outputs = pipeline.generator._format_outputs

        def timed_format_outputs(
            outputs,
            *positional,
            _original=stage_format_outputs,
            **keywords,
        ):
            def metric_value(metrics, *names):
                for name in names:
                    value = getattr(metrics, name, None)
                    if value is not None:
                        return float(value)
                return None

            batch_records = []
            generation_batch_index = len(generation_batch_records)
            for batch_position, output in enumerate(outputs):
                metrics = getattr(output, "metrics", None)
                if metrics is None:
                    continue
                queued = metric_value(metrics, "queued_ts", "queued_time")
                scheduled = metric_value(
                    metrics,
                    "first_scheduled_time",
                    "scheduled_time",
                    "scheduled_ts",
                )
                first_token = metric_value(
                    metrics, "first_token_time", "first_token_ts"
                )
                finished = metric_value(
                    metrics, "finished_time", "last_token_time", "last_token_ts"
                )
                arrival = metric_value(metrics, "arrival_time")
                first_token_latency = metric_value(metrics, "first_token_latency")
                engine_e2e = metric_value(metrics, "e2e_latency")
                if first_token is None or finished is None:
                    continue
                if scheduled is None:
                    if first_token_latency is not None:
                        scheduled = first_token - first_token_latency
                    elif arrival is not None:
                        scheduled = arrival
                    else:
                        continue
                if queued is None:
                    queued = scheduled
                queue_time = metric_value(metrics, "time_in_queue")
                if queue_time is None:
                    queue_time = max(0.0, scheduled - queued)
                prefill_service = max(0.0, first_token - scheduled)
                decode_service = max(0.0, finished - first_token)
                prompt_token_ids = list(getattr(output, "prompt_token_ids", None) or ())
                prompt_tokens = len(prompt_token_ids)
                cached_tokens = min(
                    prompt_tokens,
                    int(getattr(output, "num_cached_tokens", 0) or 0),
                )
                if first_token_latency is None:
                    first_token_latency = queue_time + prefill_service
                if engine_e2e is None:
                    engine_e2e = first_token_latency + decode_service
                record = {
                    "record_index": len(generation_stage_records),
                    "generation_batch_index": generation_batch_index,
                    "batch_position": batch_position,
                    "engine_arrival_wall_s": (
                        float(arrival) if arrival is not None else None
                    ),
                    "queued_s": float(queued),
                    "first_scheduled_s": float(scheduled),
                    "first_token_s": float(first_token),
                    "finished_s": float(finished),
                    "engine_queue_s": float(queue_time),
                    "prefill_service_s": float(prefill_service),
                    "engine_ttft_s": float(first_token_latency),
                    "decode_s": float(decode_service),
                    "generation_e2e_s": float(engine_e2e),
                    "scheduled_generation_e2e_s": float(max(0.0, finished - scheduled)),
                    "prompt_tokens": int(prompt_tokens),
                    "cached_tokens": int(cached_tokens),
                    "uncached_tokens": int(prompt_tokens - cached_tokens),
                    "prefix_hit_rate": (
                        float(cached_tokens / prompt_tokens) if prompt_tokens else 0.0
                    ),
                    "prompt_token_ids_sha256": canonical_sha256(prompt_token_ids),
                    "completion_tokens": int(
                        sum(
                            len(getattr(candidate, "token_ids", None) or ())
                            for candidate in getattr(output, "outputs", ())
                        )
                    ),
                    "finish_reasons": [
                        getattr(candidate, "finish_reason", None)
                        for candidate in getattr(output, "outputs", ())
                    ],
                    "stop_reasons": [
                        getattr(candidate, "stop_reason", None)
                        for candidate in getattr(output, "outputs", ())
                    ],
                }
                generation_stage_records.append(record)
                batch_records.append(record)
            if batch_records:
                numpy = __import__("numpy")
                tail_record = max(
                    batch_records,
                    key=lambda record: record["first_token_s"],
                )
                batch_prompt_tokens = sum(
                    record["prompt_tokens"] for record in batch_records
                )
                batch_cached_tokens = sum(
                    record["cached_tokens"] for record in batch_records
                )
                generation_batch_records.append(
                    {
                        "batch_index": generation_batch_index,
                        "request_count": len(batch_records),
                        "engine_queue_window_s": max(
                            record["first_scheduled_s"] for record in batch_records
                        )
                        - min(record["queued_s"] for record in batch_records),
                        "prefill_window_s": max(
                            record["first_token_s"] for record in batch_records
                        )
                        - min(record["first_scheduled_s"] for record in batch_records),
                        "decode_window_s": max(
                            record["finished_s"] for record in batch_records
                        )
                        - min(record["first_token_s"] for record in batch_records),
                        "generation_e2e_window_s": max(
                            record["finished_s"] for record in batch_records
                        )
                        - min(record["queued_s"] for record in batch_records),
                        "prefill_service_mean_s": statistics.fmean(
                            record["prefill_service_s"] for record in batch_records
                        ),
                        "prefill_service_p95_s": float(
                            numpy.percentile(
                                [
                                    record["prefill_service_s"]
                                    for record in batch_records
                                ],
                                95,
                            )
                        ),
                        "prefill_service_max_s": max(
                            record["prefill_service_s"] for record in batch_records
                        ),
                        "first_scheduled_skew_s": max(
                            record["first_scheduled_s"] for record in batch_records
                        )
                        - min(record["first_scheduled_s"] for record in batch_records),
                        "prompt_tokens": int(batch_prompt_tokens),
                        "cached_tokens": int(batch_cached_tokens),
                        "uncached_tokens": int(
                            batch_prompt_tokens - batch_cached_tokens
                        ),
                        "prefix_hit_rate": (
                            float(batch_cached_tokens / batch_prompt_tokens)
                            if batch_prompt_tokens
                            else 0.0
                        ),
                        "tail_request_prompt_tokens": int(tail_record["prompt_tokens"]),
                        "tail_request_cached_tokens": int(tail_record["cached_tokens"]),
                        "tail_request_uncached_tokens": int(
                            tail_record["uncached_tokens"]
                        ),
                        "tail_request_prefix_hit_rate": float(
                            tail_record["prefix_hit_rate"]
                        ),
                        "tail_request_prefill_service_s": float(
                            tail_record["prefill_service_s"]
                        ),
                    }
                )
            return _original(outputs, *positional, **keywords)

        pipeline.generator._format_outputs = timed_format_outputs
    if args.record_prefix_distribution:
        prefix_original_format_outputs = pipeline.generator._format_outputs

        def prefix_format_outputs(
            outputs,
            *positional,
            _original=prefix_original_format_outputs,
            **keywords,
        ):
            nonlocal prefix_recent_cached_tokens
            nonlocal prefix_recent_prompt_tokens
            nonlocal prefix_periodic_origin_s
            nonlocal prefix_next_periodic_sample_s
            now_s = time.perf_counter()
            if prefix_periodic_origin_s is None:
                prefix_periodic_origin_s = now_s
            for output in outputs:
                prompt_tokens = len(getattr(output, "prompt_token_ids", None) or ())
                cached_tokens = min(
                    prompt_tokens,
                    int(getattr(output, "num_cached_tokens", 0) or 0),
                )
                prefix_request_hit_rates.append(
                    cached_tokens / prompt_tokens if prompt_tokens else 0.0
                )
                prefix_request_cached_tokens.append(cached_tokens)
                prefix_request_prompt_tokens.append(prompt_tokens)
                prefix_recent_requests.append((cached_tokens, prompt_tokens))
                prefix_recent_cached_tokens += cached_tokens
                prefix_recent_prompt_tokens += prompt_tokens
                if len(prefix_recent_requests) > 1000:
                    expired_cached, expired_prompt = (
                        prefix_recent_requests.popleft()
                    )
                    prefix_recent_cached_tokens -= expired_cached
                    prefix_recent_prompt_tokens -= expired_prompt

            elapsed_s = now_s - prefix_periodic_origin_s
            if elapsed_s >= prefix_next_periodic_sample_s:
                prefix_periodic_recent_samples.append(
                    {
                        "sample_index": len(prefix_periodic_recent_samples),
                        "elapsed_s": float(elapsed_s),
                        "request_count": len(prefix_recent_requests),
                        "prompt_tokens": int(prefix_recent_prompt_tokens),
                        "cached_tokens": int(prefix_recent_cached_tokens),
                        "hit_rate": (
                            float(
                                prefix_recent_cached_tokens
                                / prefix_recent_prompt_tokens
                            )
                            if prefix_recent_prompt_tokens
                            else 0.0
                        ),
                    }
                )
                while prefix_next_periodic_sample_s <= elapsed_s:
                    prefix_next_periodic_sample_s += 1.0
            return _original(outputs, *positional, **keywords)

        pipeline.generator._format_outputs = prefix_format_outputs

    dataset = Dataset(data=rows)
    started = time.perf_counter()
    output_dataset = await poisson_persistent_workflow_run(
        dataset,
        pipeline,
        args.arrival_rate,
        workflow=args.workflow,
        do_eval=True,
        worker_count=args.worker_count if async_enabled else 1,
        microbatch_max_size=args.microbatch_size,
        arrival_burstiness=args.arrival_burstiness,
        normalize_arrival_trace=args.normalize_arrival_trace,
        random_seed=args.seed,
        arrival_offsets=arrival_offsets,
        record_request_timing=True,
    )
    wall_time = time.perf_counter() - started
    query_embedding_audit_started = time.perf_counter()
    numpy = __import__("numpy")
    canonical_query_embeddings = numpy.asarray(
        pipeline.retriever.encoder.encode(
            [str(row["question"]) for row in rows],
            batch_size=args.retrieval_batch_size,
            is_query=True,
        ),
        dtype=numpy.float32,
        order="C",
    )
    if canonical_query_embeddings.shape != (args.num_requests, 1024):
        raise RuntimeError(
            "Canonical query-embedding audit shape mismatch: "
            f"{canonical_query_embeddings.shape}"
        )
    if not numpy.all(numpy.isfinite(canonical_query_embeddings)):
        raise RuntimeError("Canonical query-embedding audit contains nonfinite values")
    query_embeddings_sha256 = array_sha256(canonical_query_embeddings)
    query_embedding_audit_s = time.perf_counter() - query_embedding_audit_started
    del canonical_query_embeddings
    if retrieval_reference_ids is not None and retrieval_reference_cursor != len(
        retrieval_reference_ids
    ):
        raise RuntimeError(
            "Retrieval reference was not consumed exactly once: "
            f"used={retrieval_reference_cursor}, "
            f"available={len(retrieval_reference_ids)}"
        )
    kv_cache_stats_total = await pipeline.generator.get_kv_cache_stats()
    kv_cache_stats = copy.deepcopy(kv_cache_stats_total)
    warmup_transfer_stats = (warmup_kv_cache_stats or {}).get("cpu_kv_transfer")
    total_transfer_stats = (kv_cache_stats_total or {}).get("cpu_kv_transfer")
    cpu_kv_transfer_latency = transfer_stats_delta(
        warmup_transfer_stats,
        total_transfer_stats,
    )
    if kv_cache_stats is not None:
        kv_cache_stats["cpu_kv_transfer"] = cpu_kv_transfer_latency
    document_heat_metadata_stats = (
        pipeline.get_document_heat_metadata_stats()
        if hasattr(pipeline, "get_document_heat_metadata_stats")
        else None
    )
    if args.cpu_kv_layout["enabled"] and cpu_kv_transfer_latency is None:
        raise RuntimeError(
            "CPU KV offload is enabled but transfer timing is unavailable"
        )
    required_directions = []
    if args.require_cpu_kv_h2d_samples:
        required_directions.append("h2d_reload")
    if args.require_cpu_kv_d2h_samples:
        required_directions.append("d2h_offload")
    for direction in required_directions:
        completed = int(
            (cpu_kv_transfer_latency or {})
            .get(direction, {})
            .get("worker_completed_jobs", 0)
        )
        if completed <= 0:
            raise RuntimeError(
                f"CPU KV {direction} timing was required but no transfer completed"
            )
    scheduler_stats = dict(pipeline.last_workflow_scheduler_stats)
    generation_admission_stats = scheduler_stats.get("generation_admission")
    if generation_admission_stats is None:
        admission_stats_getter = getattr(
            pipeline.generator, "get_async_admission_stats", None
        )
        if callable(admission_stats_getter):
            generation_admission_stats = admission_stats_getter()
    if generation_admission_stats is not None and generation_admission_stats.get(
        "permit_leak"
    ):
        raise RuntimeError("AsyncLLM generation admission permits leaked after the run")
    metric_values = collect_quality_metric_values(output_dataset, args.num_requests)
    timing_fields = (
        "scheduled_arrival_offset_s",
        "arrival_offset_s",
        "actual_arrival_offset_s",
        "arrival_schedule_lag_s",
        "finish_offset_s",
        "e2e_latency_s",
        "post_admission_latency_s",
        "scheduler_admission_wait_s",
        "scheduler_queue_wait_s",
    )
    timing_columns = {field: [] for field in timing_fields}
    retrieval_id_rows = []
    prediction_quality_rows = []
    for request_index, item in enumerate(output_dataset):
        metrics = item.output.get("metric_score") or {}
        for field in timing_fields:
            raw_value = item.output.get(field)
            if raw_value is None:
                raise RuntimeError(f"Missing {field} for request {request_index}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {field} for request {request_index}: {value}"
                )
            timing_columns[field].append(value)
        retrieval_id_rows.append(item.output.get("retrieval_idxs"))
        prediction_quality_rows.append(
            {
                "pred": item.output.get("pred"),
                "metrics": metrics,
            }
        )
    for field, values in timing_columns.items():
        if len(values) != args.num_requests:
            raise RuntimeError(
                f"Timing denominator mismatch for {field}: "
                f"expected {args.num_requests}, observed {len(values)}"
            )
    request_latencies = timing_columns["e2e_latency_s"]
    post_admission_latencies = timing_columns["post_admission_latency_s"]
    arrival_schedule_lags = timing_columns["arrival_schedule_lag_s"]
    scheduler_admission_waits = timing_columns["scheduler_admission_wait_s"]
    scheduler_queue_waits = timing_columns["scheduler_queue_wait_s"]
    for request_index in range(args.num_requests):
        scheduled = timing_columns["scheduled_arrival_offset_s"][request_index]
        arrival_alias = timing_columns["arrival_offset_s"][request_index]
        finish = timing_columns["finish_offset_s"][request_index]
        if not math.isclose(scheduled, arrival_alias, abs_tol=1e-12):
            raise RuntimeError(
                "arrival_offset_s must alias scheduled_arrival_offset_s for "
                f"request {request_index}"
            )
        if not math.isclose(
            request_latencies[request_index],
            finish - scheduled,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "E2E latency must span scheduled arrival to final completion "
                f"for request {request_index}"
            )

    runtime_monitor = getattr(args, "runtime_resource_monitor", None)
    runtime_resource_summary = None
    if runtime_monitor is not None:
        runtime_monitor.checkpoint("workload:complete")
        runtime_monitor.stop()
        runtime_monitor.assert_clean()
        runtime_resource_summary = runtime_monitor.summary()

    prompt_fingerprints = source_order_prompt_fingerprints(
        output_dataset, pipeline.generator.tokenizer
    )
    slo_summary = summarize_latency_slo(
        request_latencies,
        wall_time_s=wall_time,
        slo_seconds=args.latency_slo_seconds,
    )

    prompt_tokens = int(
        getattr(pipeline.generator, "exact_prefix_prompt_tokens", 0) or 0
    )
    cached_tokens = int(
        getattr(pipeline.generator, "exact_prefix_cached_tokens", 0) or 0
    )
    point_name = (
        f"{args.dataset}_{args.workflow}_{args.components}_"
        f"k{args.topk}_np{args.nprobe}_rps{args.arrival_rate:g}_n{args.num_requests}"
    )
    point_dir = args.output_dir / point_name
    point_dir.mkdir(parents=True, exist_ok=True)
    write_request_rows(point_dir / "requests.csv", output_dataset)
    request_metrics_path = point_dir / "request_metrics.npz"
    np = __import__("numpy")
    np.savez_compressed(
        request_metrics_path,
        sample_indices=np.asarray(sample_indices, dtype="<i8"),
        scheduled_arrival_offset_s=np.asarray(
            timing_columns["scheduled_arrival_offset_s"], dtype="<f8"
        ),
        arrival_offset_s=np.asarray(timing_columns["arrival_offset_s"], dtype="<f8"),
        actual_arrival_offset_s=np.asarray(
            timing_columns["actual_arrival_offset_s"], dtype="<f8"
        ),
        finish_offset_s=np.asarray(timing_columns["finish_offset_s"], dtype="<f8"),
        e2e_latency_s=np.asarray(request_latencies, dtype="<f8"),
        post_admission_latency_s=np.asarray(post_admission_latencies, dtype="<f8"),
        arrival_schedule_lag_s=np.asarray(arrival_schedule_lags, dtype="<f8"),
        scheduler_admission_wait_s=np.asarray(scheduler_admission_waits, dtype="<f8"),
        scheduler_queue_wait_s=np.asarray(scheduler_queue_waits, dtype="<f8"),
        retrieval_recall_at_k=np.asarray(
            metric_values["retrieval_recall"], dtype="<f4"
        ),
        f1=np.asarray(metric_values["f1"], dtype="<f4"),
        acc=np.asarray(metric_values["acc"], dtype="<f4"),
        retrieval_ids=np.asarray(retrieval_id_rows, dtype="<i8"),
        query_embeddings_sha256=np.asarray(query_embeddings_sha256),
        campaign_group_id=np.asarray(args.campaign_group_id),
        campaign_repeat=np.asarray(args.campaign_repeat, dtype="<i8"),
        campaign_order_position=np.asarray(args.campaign_order_position, dtype="<i8"),
        campaign_variant=np.asarray(args.campaign_variant),
        point_name=np.asarray(point_name),
        prefix_cached_tokens=np.asarray(prefix_request_cached_tokens, dtype="<i8"),
        prefix_prompt_tokens=np.asarray(prefix_request_prompt_tokens, dtype="<i8"),
    )
    stage_record_artifacts = None
    if args.record_stage_breakdown:
        stage_record_artifacts = {
            "generation_request": write_jsonl(
                point_dir / "generation_request_stages.jsonl",
                generation_stage_records,
            ),
            "generation_batch": write_jsonl(
                point_dir / "generation_batch_stages.jsonl",
                generation_batch_records,
            ),
            "retrieval_batch": write_jsonl(
                point_dir / "retrieval_batch_stages.jsonl",
                retrieval_batch_records,
            ),
        }
    hotness_prior_path_raw = os.environ.get("MNEMORAG_DOC_HOTNESS_PRIOR_PATH", "")
    document_hotness_prior = {
        "enabled": bool(hotness_prior_path_raw),
        "freeze": os.environ.get("MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR", "0") == "1",
    }
    if hotness_prior_path_raw:
        hotness_prior_path = Path(hotness_prior_path_raw).resolve()
        document_hotness_prior.update(
            {
                "path": str(hotness_prior_path),
                "sha256": file_sha256(hotness_prior_path),
                "size_bytes": hotness_prior_path.stat().st_size,
            }
        )
    query_base_snapshot = dict(getattr(pipeline, "query_base_snapshot", None) or {})
    query_base_snapshot_path_raw = os.environ.get(
        "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH", ""
    )
    if query_base_snapshot_path_raw:
        query_base_snapshot_path = Path(query_base_snapshot_path_raw).resolve()
        query_base_snapshot.update(
            {
                "enabled": True,
                "path": str(query_base_snapshot_path),
                "sha256": file_sha256(query_base_snapshot_path),
                "size_bytes": query_base_snapshot_path.stat().st_size,
            }
        )

    profile_next_use_path_raw = os.environ.get(
        "VLLM_KV_PROFILE_NEXT_USE_PATH",
        os.environ.get("VLLM_KV_ORACLE_NEXT_USE_PATH", ""),
    )
    profile_guided_online_serving = {
        "enabled": bool(profile_next_use_path_raw),
        "observe_only": (os.environ.get("VLLM_KV_PROFILE_OBSERVE_ONLY", "0") == "1"),
        "influences_eviction": (
            bool(profile_next_use_path_raw)
            and os.environ.get("VLLM_KV_PROFILE_OBSERVE_ONLY", "0") != "1"
        ),
        "semantics": (
            "offline_profile_observation_for_online_poisson_serving"
            if os.environ.get("VLLM_KV_PROFILE_OBSERVE_ONLY", "0") == "1"
            else "offline_profile_for_online_poisson_serving"
        ),
        "offline_profile_build_excluded_from_latency": True,
        "online_warmup_excluded_from_measurement": warmup_rows is not None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if profile_next_use_path_raw:
        profile_next_use_path = Path(profile_next_use_path_raw).resolve()
        profile_guided_online_serving.update(
            {
                "path": str(profile_next_use_path),
                "sha256": file_sha256(profile_next_use_path),
                "size_bytes": profile_next_use_path.stat().st_size,
            }
        )
    cpu_profile_path_raw = os.environ.get(
        "VLLM_CPU_KV_ADMISSION_PROFILE_PATH", ""
    )
    cpu_kv_admission_profile = {
        "enabled": bool(cpu_profile_path_raw),
        "mode": os.environ.get("VLLM_CPU_KV_PROFILE_MODE", "").strip().lower(),
    }
    if cpu_profile_path_raw:
        cpu_profile_path = Path(cpu_profile_path_raw).resolve()
        cpu_kv_admission_profile.update(
            {
                "path": str(cpu_profile_path),
                "sha256": file_sha256(cpu_profile_path),
                "size_bytes": cpu_profile_path.stat().st_size,
            }
        )

    retrieval_profile = (
        pipeline.retriever.retrieval_profile()
        if hasattr(pipeline.retriever, "retrieval_profile")
        else None
    )
    retrieval_profile_measurement = retrieval_profile_measurement_delta(
        warmup_retrieval_profile,
        retrieval_profile,
    )
    validate_single_query_probe_parallelism_contract(
        args.system_label, retrieval_profile
    )

    summary = {
        "status": "complete",
        "system": args.system_label,
        "point_name": point_name,
        "dataset": args.dataset,
        "dataset_path": str(DATASET_PATHS[args.dataset]),
        "source_queries": source_count,
        "sample_indices": sample_indices,
        "online_warmup": online_warmup,
        "online_warmup_sample_indices": warmup_sample_indices,
        "seed": args.seed,
        "workflow": args.workflow,
        "components": args.components,
        "order_mode": args.order_mode,
        "enable_similarity_aware": similarity,
        "enable_preassigned_guidance": args.preassigned_guidance,
        "enable_hotness_sort": hotness,
        "enable_async_scheduling": async_enabled,
        "kv_policy": policy,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_heat_formula": args.kv_heat_formula,
        "kv_profile_priority_mode": args.kv_profile_priority_mode,
        "kv_heat_legacy_time_scale_s": args.kv_heat_legacy_time_scale_s,
        "kv_heat_half_life_s": args.kv_heat_half_life_s,
        "kv_heat_source": args.kv_heat_source,
        "document_kv_heat": args.document_kv_heat,
        "document_heat_metadata_stats": document_heat_metadata_stats,
        "document_hotness_prior": document_hotness_prior,
        "query_base_snapshot": query_base_snapshot,
        "profile_guided_online_serving": profile_guided_online_serving,
        "cpu_kv_admission_profile": cpu_kv_admission_profile,
        "implementation_fidelity": (
            {
                "scheduler": "flashrag_native_pipeline",
                "retriever": "flashrag_cpu_faiss",
                "instrumentation_only": True,
                "headline_eligible": True,
            }
            if args.system_label == "flashrag"
            else {
                "scheduler": "mnemorag_async_pipeline",
                "retriever": "mnemorag_similarity_aware_faiss",
                "headline_eligible": True,
            }
        ),
        "num_requests": args.num_requests,
        "request_cycle_size": args.request_cycle_size,
        "arrival_rate_rps": args.arrival_rate,
        "arrival_burstiness": args.arrival_burstiness,
        "normalize_arrival_trace": args.normalize_arrival_trace,
        "arrival_trace_source": arrival_trace_source,
        "arrival_trace_manifest": arrival_manifest,
        "arrival_trace_sha256": arrival_manifest["arrival_trace_sha256"],
        "latency_slo_seconds": args.latency_slo_seconds,
        "campaign_group_id": args.campaign_group_id,
        "campaign_repeat": args.campaign_repeat,
        "campaign_order_position": args.campaign_order_position,
        "campaign_variant": args.campaign_variant,
        "topk": args.topk,
        "nprobe": args.nprobe,
        "retrieval_batch_size": args.retrieval_batch_size,
        "microbatch_size": args.microbatch_size,
        "serving_batch_size": args.microbatch_size,
        "request_batching_policy": scheduler_stats.get("request_batching_policy"),
        "preformed_work_queue": scheduler_stats.get("preformed_work_queue"),
        "cache_aware_prefill_batching": args.cache_aware_prefill_batching,
        "cache_aware_prefill_batching_stats": getattr(
            pipeline, "_cache_aware_prefill_batching_stats", None
        ),
        "max_num_batched_tokens": (args.max_num_batched_tokens or None),
        "vllm_engine_effective": vllm_engine_contract["values"],
        "vllm_engine_contract_audit": vllm_engine_contract,
        "worker_count": args.worker_count if async_enabled else 1,
        "faiss_omp_threads": config["faiss_omp_threads"],
        "faiss_search_workers": config["faiss_search_workers"],
        "single_query_probe_parallelism_enabled": config[
            "enable_single_query_probe_parallelism"
        ],
        "faiss_mmap_index": args.faiss_mmap_index,
        "drop_loaded_input_page_cache": args.drop_loaded_input_page_cache,
        "input_page_cache_release": page_cache_release,
        "retriever_preloaded_before_generator": preloaded_retriever is not None,
        "retriever_preloaded_before_cpu_kv": bool(
            args.cpu_kv_layout["enabled"] and preloaded_retriever is not None
        ),
        "startup_gpu_reservation": getattr(
            args, "startup_gpu_reservation_report", None
        ),
        "stage_record_artifacts": stage_record_artifacts,
        "request_metrics_artifact": {
            "path": str(request_metrics_path.resolve()),
            "sha256": file_sha256(request_metrics_path),
            "request_count": args.num_requests,
        },
        "async_retrieval_policy": args.async_retrieval_policy,
        "async_retrieval_policy_effective": scheduler_stats.get(
            "effective_async_retrieval_policy",
            "eager" if args.direct_vllm_dispatch else args.async_retrieval_policy,
        ),
        "direct_vllm_dispatch": args.direct_vllm_dispatch,
        "application_generation_gate": scheduler_stats.get(
            "application_generation_gate", not args.direct_vllm_dispatch
        ),
        "cross_microbatch_pipeline": args.cross_microbatch_pipeline,
        "cross_microbatch_pipeline_effective": scheduler_stats.get(
            "cross_microbatch_pipeline"
        ),
        "retrieval_batcher": scheduler_stats.get("retrieval_batcher"),
        "query_base_exact_lookup": bool(
            getattr(pipeline.query_pool, "enable_exact_lookup", False)
        ),
        "native_first_token_event": bool(
            getattr(pipeline.generator, "use_native_first_token_event", False)
        ),
        "native_prefill_admission_event": bool(
            getattr(
                pipeline.generator,
                "use_native_prefill_start_event",
                False,
            )
        ),
        "generation_async_request_count": int(
            getattr(pipeline.generator, "async_request_count", 0)
        ),
        "generation_stream_update_count": int(
            getattr(pipeline.generator, "async_stream_update_count", 0)
        ),
        "generation_admission": generation_admission_stats,
        "retrieval_reference_replay": retrieval_reference_ids is not None,
        "retrieval_reference_csv": (
            str(args.retrieval_reference_csv.resolve())
            if args.retrieval_reference_csv is not None
            else None
        ),
        "retrieval_reference_sha256": retrieval_reference_sha256,
        "retrieval_search_executed": True,
        "retrieval_ids_sha256": canonical_sha256(retrieval_id_rows),
        "retrieval_id_sets_sha256": canonical_sha256(
            canonical_retrieval_id_sets(retrieval_id_rows)
        ),
        "prediction_quality_sha256": canonical_sha256(prediction_quality_rows),
        "generation_prompt_execution_token_ids_sha256": canonical_sha256(
            [record["prompt_token_ids_sha256"] for record in generation_stage_records]
        )
        if generation_stage_records
        else None,
        "generation_prompt_execution_lengths_sha256": canonical_sha256(
            [record["prompt_tokens"] for record in generation_stage_records]
        )
        if generation_stage_records
        else None,
        "generation_prompt_token_ids_sha256": prompt_fingerprints[
            "prompt_token_ids_sha256"
        ],
        "generation_prompt_lengths_sha256": prompt_fingerprints[
            "prompt_lengths_sha256"
        ],
        "generation_prompt_source_order": prompt_fingerprints,
        "async_fixed_delay_ms": args.async_fixed_delay_ms,
        "async_adaptive_ema_alpha": args.async_adaptive_ema_alpha,
        "async_adaptive_max_prefill_overlap_fraction": (
            args.async_adaptive_max_prefill_overlap_fraction
        ),
        "async_adaptive_safety_margin_ms": args.async_adaptive_safety_margin_ms,
        "tau_reuse": args.tau_reuse,
        "tau_guided": args.tau_guided,
        "similarity_nprobe_budget": {
            "minimum_fraction": args.similarity_nprobe_min_fraction,
            "maximum_fraction": args.similarity_nprobe_max_fraction,
            "minimum": similarity_nprobe_min,
            "maximum_query_budget": similarity_nprobe_max,
            "gamma": args.similarity_nprobe_gamma,
            "curve": args.similarity_nprobe_curve,
        },
        "irg_rounds": args.irg_rounds,
        "max_tokens": args.max_tokens,
        "recomp_compression_max_tokens": 128,
        "temperature": 0,
        "ignore_eos": args.ignore_eos,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "prefix_caching": args.prefix_caching,
        "model_path": str(ASSET_ROOT / "models/llama3-8B-instruct"),
        "retrieval_model_path": str(ASSET_ROOT / "models/e5-large-v2"),
        "encoder_contract": encoder_contract,
        "query_embeddings_sha256": query_embeddings_sha256,
        "query_embedding_audit_s": query_embedding_audit_s,
        "index_path": str(ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"),
        "corpus_path": str(ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"),
        "cpu_kv_offload": args.cpu_kv_layout,
        "physical_gpu_id": args.gpu_id,
        "visible_gpu_count": 1,
        "tensor_parallel_size": 1,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "cpu_topology": args.cpu_topology,
        "numa_policy": args.numa_policy,
        "numa_pages_before": args.numa_pages_before,
        "numa_pages_after": current_numa_page_counts(),
        "resource_preflight": args.resource_preflight,
        "runtime_resource_monitor": runtime_resource_summary,
        "initialization_profile": {
            "retriever_preload_s": retriever_preload_s,
            "pipeline_wall_s": pipeline_init_s,
            "pipeline": getattr(pipeline, "initialization_profile", None),
        },
        "wall_time_s": wall_time,
        "throughput_qps": args.num_requests / wall_time,
        "request_latency_mean_s": statistics.fmean(request_latencies),
        "request_latency_p50_s": float(
            __import__("numpy").percentile(request_latencies, 50)
        ),
        "request_latency_p95_s": float(
            __import__("numpy").percentile(request_latencies, 95)
        ),
        "request_latency_p99_s": float(
            __import__("numpy").percentile(request_latencies, 99)
        ),
        "latency_accounting": {
            "canonical_origin": "scheduled_arrival",
            "completion": "per_request_final_generation",
            "includes_arrival_schedule_lag": True,
            "post_admission_metric": "post_admission_latency_s",
        },
        "post_admission_latency_mean_s": (
            statistics.fmean(post_admission_latencies)
            if post_admission_latencies
            else 0.0
        ),
        "post_admission_latency_p95_s": (
            float(__import__("numpy").percentile(post_admission_latencies, 95))
            if post_admission_latencies
            else 0.0
        ),
        "arrival_schedule_lag_mean_s": (
            statistics.fmean(arrival_schedule_lags) if arrival_schedule_lags else 0.0
        ),
        "arrival_schedule_lag_p95_s": (
            float(__import__("numpy").percentile(arrival_schedule_lags, 95))
            if arrival_schedule_lags
            else 0.0
        ),
        "arrival_schedule_lag_max_s": (
            max(arrival_schedule_lags) if arrival_schedule_lags else 0.0
        ),
        **slo_summary,
        "scheduler_admission_wait_mean_s": (
            statistics.fmean(scheduler_admission_waits)
            if scheduler_admission_waits
            else 0.0
        ),
        "scheduler_admission_wait_p50_s": (
            float(__import__("numpy").percentile(scheduler_admission_waits, 50))
            if scheduler_admission_waits
            else 0.0
        ),
        "scheduler_admission_wait_p95_s": (
            float(__import__("numpy").percentile(scheduler_admission_waits, 95))
            if scheduler_admission_waits
            else 0.0
        ),
        "scheduler_queue_wait_mean_s": (
            statistics.fmean(scheduler_queue_waits) if scheduler_queue_waits else 0.0
        ),
        "scheduler_queue_wait_p95_s": (
            float(__import__("numpy").percentile(scheduler_queue_waits, 95))
            if scheduler_queue_waits
            else 0.0
        ),
        "prefix_cached_tokens": cached_tokens,
        "prefix_prompt_tokens": prompt_tokens,
        "prefix_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "online_prefix_cache_windows": online_prefix_cache_windows(
            generation_stage_records,
            window_seconds=10.0,
        ),
        "online_prefix_cache_recent_requests": (
            online_prefix_cache_recent_request_window(
                generation_stage_records,
                requested_window_size=1000,
            )
        ),
        "online_prefix_cache_periodic_recent_requests": {
            "metric_definition": (
                "one-second samples of the token-weighted cache-hit rate over "
                "the most recent 1000 completed requests"
            ),
            "sample_interval_s": 1.0,
            "requested_window_size": 1000,
            "sample_count": len(prefix_periodic_recent_samples),
            "mean_hit_rate": (
                statistics.fmean(
                    sample["hit_rate"] for sample in prefix_periodic_recent_samples
                )
                if prefix_periodic_recent_samples
                else None
            ),
            "max_hit_rate": (
                max(
                    sample["hit_rate"] for sample in prefix_periodic_recent_samples
                )
                if prefix_periodic_recent_samples
                else None
            ),
            "tail_hit_rate": (
                prefix_periodic_recent_samples[-1]["hit_rate"]
                if prefix_periodic_recent_samples
                else None
            ),
            "samples": prefix_periodic_recent_samples,
        },
        "kv_cache_stats": kv_cache_stats,
        "kv_cache_stats_total_including_warmup": kv_cache_stats_total,
        "cpu_kv_transfer_latency": cpu_kv_transfer_latency,
        "metrics": {
            (
                f"retrieval_recall_top{args.topk}"
                if name == "retrieval_recall"
                else name
            ): statistics.fmean(values) if values else None
            for name, values in metric_values.items()
        },
        "metric_observations": {
            (
                f"retrieval_recall_top{args.topk}"
                if name == "retrieval_recall"
                else name
            ): len(values)
            for name, values in metric_values.items()
        },
        "metric_contract": {
            "version": 2,
            "retrieval": {
                "name": "answer_hit_recall_at_k",
                "k": args.topk,
                "scope": "final_retrieval_round",
                "denominator": "all_requests",
            },
            "generation": ["f1", "acc"],
        },
        "prompt_contract": expected_prompt_contract(
            max_model_len=args.max_model_len,
            answer_max_tokens=args.max_tokens,
            irg_round_max_tokens=args.max_tokens,
            recomp_compression_max_tokens=128,
            ordering_mode=args.order_mode,
        ),
        "scheduler": scheduler_stats,
        "batch_stage_scheduler": getattr(pipeline, "last_async_scheduler_stats", None),
        "batch_stage_trace": getattr(pipeline, "last_async_scheduler_trace", None),
        "retrieval_profile": retrieval_profile,
        "retrieval_profile_measurement": retrieval_profile_measurement,
        "versions": package_versions(),
        "cache_isolation": "fresh process and fresh AsyncLLM engine per point",
    }
    if args.record_stage_breakdown:
        if (
            not retrieval_latencies_s
            or not retrieval_batch_records
            or not generation_stage_records
            or not generation_batch_records
        ):
            raise RuntimeError(
                "Stage breakdown requested but timing records are missing"
            )

        def distribution(values):
            values = [float(value) for value in values]
            if not values:
                return {"count": 0, "mean_s": None, "p50_s": None, "p95_s": None}
            numpy = __import__("numpy")
            return {
                "count": len(values),
                "mean_s": statistics.fmean(values),
                "p50_s": float(numpy.percentile(values, 50)),
                "p95_s": float(numpy.percentile(values, 95)),
            }

        retrieval_phase_names = sorted(
            {name for record in retrieval_batch_records for name in record["phase_s"]}
        )
        retrieval_batch_phases = {
            name: distribution(
                record["phase_s"].get(name, 0.0) for record in retrieval_batch_records
            )
            for name in retrieval_phase_names
        }
        retrieval_request_phases = {
            name: distribution(
                record["phase_s"].get(name, 0.0)
                for record in retrieval_batch_records
                for _ in range(int(record["request_count"]))
            )
            for name in retrieval_phase_names
        }

        scheduler_trace = list(
            getattr(pipeline, "last_async_scheduler_trace", None) or ()
        )

        def trace_interval(record, start_name, end_name):
            start = record.get(start_name)
            end = record.get(end_name)
            if start is None or end is None:
                return None
            return max(0.0, float(end) - float(start))

        trace_stage_boundaries = {
            "post_retrieval_preparation": (
                "retrieval_end_s",
                "prompt_ready_s",
            ),
            "generation_admission": (
                "generation_dispatch_s",
                "generation_start_s",
            ),
            "prefill": ("generation_start_s", "decode_ready_s"),
            "decode_partition": ("decode_ready_s", "generation_end_s"),
            "pipeline_e2e": ("retrieval_start_s", "generation_end_s"),
        }
        trace_batch_stages = {}
        trace_request_stages = {}
        for name, (start_name, end_name) in trace_stage_boundaries.items():
            batch_values = []
            request_values = []
            for record in scheduler_trace:
                value = trace_interval(record, start_name, end_name)
                if value is None:
                    continue
                batch_values.append(value)
                request_values.extend(
                    [value] * max(1, int(record.get("batch_size", 1)))
                )
            trace_batch_stages[name] = distribution(batch_values)
            trace_request_stages[name] = distribution(request_values)

        request_stage_distributions = {
            name: distribution(record[name] for record in generation_stage_records)
            for name in (
                "engine_queue_s",
                "prefill_service_s",
                "engine_ttft_s",
                "decode_s",
                "generation_e2e_s",
                "scheduled_generation_e2e_s",
            )
        }
        request_stage_distributions["system_e2e_s"] = distribution(request_latencies)
        batch_stage_distributions = {
            name: distribution(record[name] for record in generation_batch_records)
            for name in (
                "engine_queue_window_s",
                "prefill_window_s",
                "decode_window_s",
                "generation_e2e_window_s",
            )
        }
        batch_stage_distributions["retrieval_total_s"] = distribution(
            record["total_s"] for record in retrieval_batch_records
        )
        request_stage_distributions["retrieval_total_s"] = distribution(
            retrieval_latencies_s
        )

        def pearson(x_values, y_values):
            numpy = __import__("numpy")
            x = numpy.asarray(list(x_values), dtype=float)
            y = numpy.asarray(list(y_values), dtype=float)
            if len(x) < 2 or numpy.ptp(x) == 0 or numpy.ptp(y) == 0:
                return None
            return float(numpy.corrcoef(x, y)[0, 1])

        cache_hit_records = [
            record for record in generation_stage_records if record["cached_tokens"] > 0
        ]
        cache_miss_records = [
            record
            for record in generation_stage_records
            if record["cached_tokens"] == 0
        ]
        prefix_prefill_analysis = {
            "prompt_tokens": distribution(
                record["prompt_tokens"] for record in generation_stage_records
            ),
            "cached_tokens": distribution(
                record["cached_tokens"] for record in generation_stage_records
            ),
            "uncached_tokens": distribution(
                record["uncached_tokens"] for record in generation_stage_records
            ),
            "request_prefix_hit_rate": distribution(
                record["prefix_hit_rate"] for record in generation_stage_records
            ),
            "prefill_by_cache_presence": {
                "cache_hit": distribution(
                    record["prefill_service_s"] for record in cache_hit_records
                ),
                "cache_miss": distribution(
                    record["prefill_service_s"] for record in cache_miss_records
                ),
            },
            "correlation": {
                "cached_tokens_vs_prefill_service": pearson(
                    (record["cached_tokens"] for record in generation_stage_records),
                    (
                        record["prefill_service_s"]
                        for record in generation_stage_records
                    ),
                ),
                "uncached_tokens_vs_prefill_service": pearson(
                    (record["uncached_tokens"] for record in generation_stage_records),
                    (
                        record["prefill_service_s"]
                        for record in generation_stage_records
                    ),
                ),
                "prefix_hit_rate_vs_prefill_service": pearson(
                    (record["prefix_hit_rate"] for record in generation_stage_records),
                    (
                        record["prefill_service_s"]
                        for record in generation_stage_records
                    ),
                ),
            },
            "batch_tail": {
                "tail_request_prefix_hit_rate": distribution(
                    record["tail_request_prefix_hit_rate"]
                    for record in generation_batch_records
                ),
                "tail_request_uncached_tokens": distribution(
                    record["tail_request_uncached_tokens"]
                    for record in generation_batch_records
                ),
                "tail_request_prefill_service_s": distribution(
                    record["tail_request_prefill_service_s"]
                    for record in generation_batch_records
                ),
                "first_scheduled_skew_s": distribution(
                    record["first_scheduled_skew_s"]
                    for record in generation_batch_records
                ),
            },
        }

        h2d = (cpu_kv_transfer_latency or {}).get("h2d_reload", {})
        summary["stage_latency"] = {
            "retrieval_request_count": len(retrieval_latencies_s),
            "generation_request_count": len(generation_stage_records),
            "retrieval_batch_count": len(retrieval_batch_records),
            "generation_batch_count": len(generation_batch_records),
            "retrieval_mean_s": statistics.fmean(retrieval_latencies_s),
            "retrieval_p95_s": float(
                __import__("numpy").percentile(retrieval_latencies_s, 95)
            ),
            "generation_e2e_mean_s": statistics.fmean(
                record["generation_e2e_s"]
                for record in generation_stage_records
                if record["generation_e2e_s"] is not None
            ),
            "prefill_mean_s": statistics.fmean(
                record["prefill_service_s"] for record in generation_stage_records
            ),
            "decode_mean_s": statistics.fmean(
                record["decode_s"] for record in generation_stage_records
            ),
            "batch": {
                "stages": batch_stage_distributions,
                "scheduler_trace_stages": trace_batch_stages,
                "retrieval_phases": retrieval_batch_phases,
                "h2d_transfer_job": {
                    "count": int(h2d.get("worker_completed_jobs", 0)),
                    "mean_s": float(h2d.get("worker_observed_mean_s", 0.0)),
                    "p95_s": float(h2d.get("worker_observed_p95_recent_s", 0.0)),
                },
            },
            "request": {
                "stages": request_stage_distributions,
                "prefix_prefill_analysis": prefix_prefill_analysis,
                "scheduler_trace_stages": trace_request_stages,
                "retrieval_phases": retrieval_request_phases,
                "h2d_request_wait": {
                    "count": int(h2d.get("scheduler_wait_completed_requests", 0)),
                    "mean_s": float(h2d.get("scheduler_wait_mean_s", 0.0)),
                    "max_s": float(h2d.get("scheduler_wait_max_s", 0.0)),
                },
            },
            "measurement": (
                "Batch metrics are unweighted wall-clock stage windows. Request "
                "metrics use per-request vLLM timestamps; shared retrieval and "
                "scheduler intervals are request-weighted without dividing by "
                "batch size. Prefill service is first_scheduled_time to "
                "first_token_time; engine TTFT includes engine queueing. Stage "
                "windows may overlap and are not additive."
            ),
        }
    if args.record_prefix_distribution:
        if not prefix_request_hit_rates:
            raise RuntimeError(
                "Prefix distribution requested but per-request records are missing"
            )
        summary["prefix_request_hit_rate_mean"] = statistics.fmean(
            prefix_request_hit_rates
        )
        summary["prefix_request_hit_rate_p50"] = float(
            __import__("numpy").percentile(prefix_request_hit_rates, 50)
        )
        summary["prefix_request_hit_rate_p95"] = float(
            __import__("numpy").percentile(prefix_request_hit_rates, 95)
        )
        summary["prefix_request_hit_rate_max"] = max(prefix_request_hit_rates)
        summary["prefix_request_count"] = len(prefix_request_hit_rates)
    write_json(point_dir / "summary.json", summary)
    print(
        "POINT_COMPLETE "
        + json.dumps(
            {
                "point_name": point_name,
                "wall_time_s": wall_time,
                "throughput_qps": summary["throughput_qps"],
                "request_latency_mean_s": summary["request_latency_mean_s"],
                "request_latency_p95_s": summary["request_latency_p95_s"],
                "prefix_hit_rate": summary["prefix_hit_rate"],
                "metrics": summary["metrics"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    pipeline.close()
    return summary


def main() -> None:
    args = parse_args()
    selected_cpus = configure_resources(args)
    reservation = start_startup_gpu_reservation(args.startup_gpu_reservation_mib)
    args.startup_gpu_reservation_handle = reservation
    try:
        monitor = RuntimeResourceMonitor(
            selected_cpus,
            args.gpu_id,
            require_clean=args.require_clean_resources,
            max_external_cpu_busy_percent=(args.max_runtime_external_cpu_busy_percent),
            sample_interval_s=args.runtime_resource_sample_seconds,
        )
        args.runtime_resource_monitor = monitor
        monitor.start()
        try:
            asyncio.run(run(args, selected_cpus))
        finally:
            monitor.stop()
    finally:
        args.startup_gpu_reservation_report = release_startup_gpu_reservation(
            reservation
        )


if __name__ == "__main__":
    main()
