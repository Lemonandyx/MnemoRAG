#!/usr/bin/env python3
"""Reproduce the Figure-14 NQ top-k=3 contract with both heat formulas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

try:
    from .build_next_use_profile import build_profiles
except ImportError:
    from build_next_use_profile import build_profiles


REQUESTS = 79_168
TARGET_CUMULATIVE_HIT_RATE = 0.303006
TARGET_ABSOLUTE_TOLERANCE = 0.02
FORMULAS = ("exponential_decay", "legacy_log_denominator")


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-point", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--workflow-utils-root",
        type=Path,
        required=True,
        help="Directory containing the audited online-workflow utils.py.",
    )
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-list", default="0-63")
    parser.add_argument("--numa-nodes", default="0-3")
    parser.add_argument("--requests", type=int, default=REQUESTS)
    parser.add_argument(
        "--cpu-kv-pinned-tier-gib",
        type=float,
        default=64.0,
        help=(
            "Physical pinned tier for the exact 130,000-block CPU KV pool; "
            "the remainder uses lazy mmap backing."
        ),
    )
    parser.add_argument(
        "--cpu-kv-lazy-mmap-dir",
        type=Path,
        default=Path("/dev/shm"),
        help="Existing tmpfs or disk-backed directory for the lazy CPU KV tier.",
    )
    parser.add_argument(
        "--target-hit-rate", type=float, default=TARGET_CUMULATIVE_HIT_RATE
    )
    parser.add_argument(
        "--target-tolerance", type=float, default=TARGET_ABSOLUTE_TOLERANCE
    )
    parser.add_argument("--reuse-capture", action="store_true")
    return parser.parse_args()


def point_dir(output_dir: Path, requests: int) -> Path:
    return output_dir / f"nq_onepass_full_k3_np128_rps24_n{requests}"


def common_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        "numactl",
        f"--physcpubind={args.cpu_list}",
        f"--interleave={args.numa_nodes}",
        str(args.python),
        str(args.benchmark_point),
        "--dataset", "nq",
        "--workflow", "onepass",
        "--num-requests", str(args.requests),
        "--arrival-rate", "24",
        "--arrival-burstiness", "1",
        "--normalize-arrival-trace",
        "--latency-slo-seconds", "10",
        "--campaign-group-id", "full_nq_heat_contract",
        "--seed", "0",
        "--topk", "3",
        "--nprobe", "128",
        "--max-model-len", "2048",
        "--max-tokens", "32",
        "--gpu-memory-utilization", "0.90",
        "--gpu-id", str(args.gpu_id),
        "--kv-cache-dtype", "fp8",
        "--cpu-threads", "64",
        "--require-clean-resources",
        "--max-preflight-cpu-busy-percent", "100",
        "--max-preflight-cpu-max-busy-percent", "100",
        "--max-preflight-gpu-memory-mib", "256",
        "--max-preflight-gpu-util-percent", "5",
        "--runtime-resource-sample-seconds", "10",
        "--max-runtime-external-cpu-busy-percent", "100",
        "--output-dir", str(output_dir),
        "--components", "full",
        "--order-mode", "hotness_hint",
        "--retrieval-batch-size", "64",
        "--microbatch-size", "64",
        "--worker-count", "8",
        "--async-retrieval-policy", "eager",
        "--direct-vllm-dispatch",
        "--max-num-batched-tokens", "12288",
        "--faiss-omp-threads", "1",
        "--faiss-search-workers", "64",
        "--drop-loaded-input-page-cache",
        "--prefix-caching",
        "--record-prefix-distribution",
        "--cache-aware-prefill-batching",
        "--no-cross-microbatch-pipeline",
        "--kv-policy-override", "heat",
        "--kv-heat-source", "document",
        "--kv-profile-priority-mode", "formula_tiebreak",
    ]


def clean_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    python_path = [
        str(args.project_root),
        str(args.project_root / "benchmarks"),
        str(args.project_root / "experiments/mnemorag_evaluation"),
        str(args.workflow_utils_root),
        str(args.vllm_root),
    ]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(python_path),
            "MNEMORAG_VLLM_ROOT": str(args.vllm_root),
            "MNEMORAG_ASSET_ROOT": str(args.asset_root),
            "MNEMORAG_PROJECT_ROOT": str(args.project_root),
        }
    )
    for name in (
        "VLLM_KV_PROFILE_NEXT_USE_PATH",
        "VLLM_KV_ORACLE_NEXT_USE_PATH",
        "VLLM_KV_PROFILE_OBSERVE_ONLY",
        "VLLM_KV_PROFILE_RECORD_PATH",
        "VLLM_CPU_KV_ADMISSION_PROFILE_PATH",
        "VLLM_CPU_KV_PROFILE_MODE",
        "VLLM_KV_HEAT_LEGACY_TIMESTAMP_SCORE",
    ):
        environment.pop(name, None)
    return environment


def run_logged(
    command: list[str], environment: dict[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"command failed with status {return_code}; inspect {log_path}"
        )


def load_summary(output_dir: Path, requests: int) -> dict:
    summary_path = point_dir(output_dir, requests) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def recovered_capture_manifest(
    access_log: Path, run_log: Path, requests: int
) -> dict[str, object]:
    positions: set[int] = set()
    recorded_lines = 0
    maximum_position = None
    with access_log.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            position = int(record["request_position"])
            if position < 0 or position >= requests:
                raise RuntimeError(
                    f"capture position out of range on line {line_number}"
                )
            positions.add(position)
            maximum_position = (
                position
                if maximum_position is None
                else max(maximum_position, position)
            )
            recorded_lines += 1
    run_text = run_log.read_text(encoding="utf-8", errors="replace")
    completed_marker = f"requests={requests}," in run_text
    coverage = len(positions) / requests
    manifest = {
        "format": "mnemorag.profile-capture-recovery.v1",
        "status": "complete" if completed_marker else "failed",
        "reason": "post-workload evaluator artifact path failed before summary",
        "run_log": str(run_log.resolve()),
        "run_log_sha256": file_sha256(run_log),
        "access_log": str(access_log.resolve()),
        "access_log_sha256": file_sha256(access_log),
        "recorded_lines": recorded_lines,
        "distinct_request_positions": len(positions),
        "request_position_coverage": coverage,
        "maximum_request_position": maximum_position,
        "persistent_workflow_completed_marker": completed_marker,
    }
    if (
        not completed_marker
        or maximum_position != requests - 1
        or coverage < 0.99
        or recorded_lines <= 0
    ):
        raise RuntimeError(f"incomplete recovered profile capture: {manifest}")
    return manifest


def capture_profile(args: argparse.Namespace) -> tuple[Path, Path | None]:
    capture_dir = args.output_root / "profile_capture"
    access_log = args.output_root / "profiles" / "block_accesses.jsonl"
    access_log.parent.mkdir(parents=True, exist_ok=True)
    summary_path = point_dir(capture_dir, args.requests) / "summary.json"
    run_log = capture_dir / "run.log"
    if args.reuse_capture and access_log.is_file() and not summary_path.is_file():
        recovery = recovered_capture_manifest(access_log, run_log, args.requests)
        atomic_write_json(
            args.output_root / "profiles/capture_recovery.json", recovery
        )
        return access_log, None
    if not (args.reuse_capture and summary_path.is_file() and access_log.is_file()):
        if access_log.exists():
            raise FileExistsError(
                f"refusing to append a second trace to {access_log}"
            )
        environment = clean_environment(args)
        environment["VLLM_KV_PROFILE_RECORD_PATH"] = str(access_log)
        command = common_command(args, capture_dir) + [
            "--kv-heat-formula", "exponential_decay",
            "--cpu-kv-cache-gib", "0",
            "--cpu-kv-num-blocks", "0",
            "--cpu-kv-pinned-tier-gib", "0",
        ]
        run_logged(command, environment, run_log)
    capture = load_summary(capture_dir, args.requests)
    if capture.get("source_queries") != REQUESTS:
        raise RuntimeError("profile capture is not the complete 79,168-row NQ")
    if int((capture.get("kv_cache_stats") or {}).get("profile_record_count", 0)) <= 0:
        raise RuntimeError("profile capture did not record KV block accesses")
    return access_log, point_dir(capture_dir, args.requests) / "requests.csv"


def validate_formula_summary(
    summary: dict,
    formula: str,
    requests: int,
    target: float,
    tolerance: float,
) -> dict[str, object]:
    stats = summary.get("kv_cache_stats") or {}
    transfers = stats.get("cpu_kv_transfer") or {}
    h2d = transfers.get("h2d_reload") or {}
    d2h = transfers.get("d2h_offload") or {}
    rate = float(summary.get("prefix_hit_rate", -1.0))
    cpu_profile = summary.get("cpu_kv_admission_profile") or {}
    cpu_offload = summary.get("cpu_kv_offload") or {}
    match_count = int(stats.get("profile_next_use_matches", 0))
    access_count = int(stats.get("profile_next_use_accesses", 0))
    checks = {
        "status_complete": summary.get("status") == "complete",
        "complete_nq": summary.get("source_queries") == REQUESTS
        and summary.get("num_requests") == requests
        and len(summary.get("sample_indices") or ()) == requests,
        "topk3_qps24": summary.get("topk") == 3
        and float(summary.get("arrival_rate_rps", 0.0)) == 24.0,
        "fp8_gpu090_seq64": summary.get("kv_cache_dtype") == "fp8"
        and float(summary.get("gpu_memory_utilization", 0.0)) == 0.9
        and summary.get("vllm_engine_effective", {}).get("max_num_seqs") == 64,
        "cpu_130000_blocks": cpu_offload.get("num_cpu_blocks") == 130_000
        and cpu_offload.get("capacity_source") == "exact_block_count"
        and cpu_offload.get("kv_cache_dtype") == "fp8"
        and cpu_offload.get("block_bytes") == 1_048_576,
        "cpu_lazy_backing_audited": cpu_offload.get("host_memory_mode")
        == "hybrid_pinned_lazy"
        and cpu_offload.get("lazy_allocation") in {
            "tmpfs_mmap",
            "file_mmap",
        }
        and bool(cpu_offload.get("lazy_mmap_dir"))
        and int(cpu_offload.get("lazy_backing_required_bytes", 0)) > 0,
        "cpu_productive_profile": cpu_profile.get("enabled") is True
        and cpu_profile.get("mode") == ""
        and bool(cpu_profile.get("sha256")),
        "selected_formula": stats.get("heat_formula") == formula,
        "legacy_time_normalized": float(
            stats.get("legacy_time_scale_s", 0.0)
        ) == 1.0,
        "formula_executed": int(stats.get("heat_formula_priority_calls", 0)) > 0,
        "profile_formula_composed": stats.get("profile_priority_mode")
        == "formula_tiebreak"
        and int(stats.get("profile_formula_tiebreak_calls", 0)) > 0
        and int(stats.get("profile_formula_tiebreak_evictions", 0)) > 0,
        "profile_matches_trace": access_count > 0
        and match_count / access_count >= 0.99,
        "offload_active": int(h2d.get("worker_transferred_blocks", 0)) > 0
        and int(d2h.get("worker_transferred_blocks", 0)) > 0,
        "figure14_target": abs(rate - target) <= tolerance,
        "legacy_zero_guard": formula != "legacy_log_denominator"
        or int(stats.get("legacy_zero_guard_calls", 0)) > 0,
    }
    return {
        "formula": formula,
        "prefix_hit_rate": rate,
        "target": target,
        "absolute_tolerance": tolerance,
        "delta": rate - target,
        "profile_match_fraction": (
            match_count / access_count if access_count else None
        ),
        "formula_priority_calls": stats.get("heat_formula_priority_calls"),
        "profile_formula_tiebreak_evictions": stats.get(
            "profile_formula_tiebreak_evictions"
        ),
        "legacy_zero_guard_calls": stats.get("legacy_zero_guard_calls"),
        "h2d_blocks": h2d.get("worker_transferred_blocks"),
        "d2h_blocks": d2h.get("worker_transferred_blocks"),
        "gpu_kv_blocks": stats.get("num_gpu_blocks"),
        "cpu_kv_admission_profile": cpu_profile,
        "recent_1000": summary.get("online_prefix_cache_recent_requests"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    args.cpu_kv_lazy_mmap_dir.mkdir(parents=True, exist_ok=True)
    if args.requests != REQUESTS:
        raise ValueError("the formal contract requires all 79,168 NQ requests")
    if args.target_tolerance <= 0:
        raise ValueError("--target-tolerance must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    # The public evaluator writes this legacy relative artifact before the
    # benchmark serializes its authoritative point summary.
    (args.output_root / "output").mkdir(parents=True, exist_ok=True)

    access_log, retrieval_reference = capture_profile(args)
    profiles_dir = args.output_root / "profiles"
    gpu_profile = profiles_dir / "gpu_next_use.json"
    cpu_profile = profiles_dir / "cpu_productive_admission.json"
    profile_manifest = profiles_dir / "manifest.json"
    build_profiles(
        access_log, gpu_profile, cpu_profile, profile_manifest
    )

    summaries: dict[str, dict] = {}
    validations: dict[str, dict] = {}
    for order, formula in enumerate(FORMULAS):
        output_dir = args.output_root / formula
        summary_path = point_dir(output_dir, args.requests) / "summary.json"
        if not summary_path.is_file():
            environment = clean_environment(args)
            environment.update(
                {
                    "VLLM_KV_PROFILE_NEXT_USE_PATH": str(gpu_profile),
                    "VLLM_KV_PROFILE_OBSERVE_ONLY": "0",
                    "VLLM_CPU_KV_ADMISSION_PROFILE_PATH": str(cpu_profile),
                    "VLLM_CPU_KV_PROFILE_MODE": "",
                }
            )
            command = common_command(args, output_dir) + [
                "--campaign-order-position", str(order),
                "--kv-heat-formula", formula,
                "--cpu-kv-cache-gib", str(130_000 * 1_048_576 / 1024**3),
                "--cpu-kv-num-blocks", "130000",
                "--cpu-kv-block-size", "16",
                "--cpu-kv-eviction-policy", "lru",
                "--cpu-kv-batch-h2d-transfers",
                "--cpu-kv-batch-d2h-transfers",
                "--cpu-kv-h2d-scheduling", "async",
                "--cpu-kv-document-prefix-only",
                "--cpu-kv-min-document-heat-to-store", "1",
                "--cpu-kv-h2d-stream-priority", "-1",
                "--cpu-kv-d2h-stream-priority", "0",
                "--cpu-kv-host-memory-mode", "hybrid_pinned_lazy",
                "--cpu-kv-pinned-tier-gib", str(args.cpu_kv_pinned_tier_gib),
                "--cpu-kv-lazy-mmap-dir", str(args.cpu_kv_lazy_mmap_dir),
                "--cpu-kv-staging-buffer-blocks", "1024",
                "--require-cpu-kv-h2d-samples",
                "--require-cpu-kv-d2h-samples",
            ]
            if retrieval_reference is not None:
                command.extend(
                    ["--retrieval-reference-csv", str(retrieval_reference)]
                )
            run_logged(command, environment, output_dir / "run.log")
        summary = load_summary(output_dir, args.requests)
        summaries[formula] = summary
        validations[formula] = validate_formula_summary(
            summary,
            formula,
            args.requests,
            args.target_hit_rate,
            args.target_tolerance,
        )

    formula_delta = abs(
        validations[FORMULAS[0]]["prefix_hit_rate"]
        - validations[FORMULAS[1]]["prefix_hit_rate"]
    )
    cross_formula_check = formula_delta <= 0.005
    aggregate = {
        "format": "mnemorag.full-nq-heat-contract.v1",
        "status": (
            "complete"
            if all(item["passed"] for item in validations.values())
            and cross_formula_check
            else "failed"
        ),
        "contract": {
            "dataset": "NQ",
            "requests": REQUESTS,
            "topk": 3,
            "arrival_rate_rps": 24,
            "gpu_id": str(args.gpu_id),
            "physical_cpu_cores": 64,
            "gpu_kv_dtype": "fp8",
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 64,
            "cpu_kv_blocks": 130_000,
            "target_cumulative_hit_rate": args.target_hit_rate,
            "target_absolute_tolerance": args.target_tolerance,
            "profile_priority_mode": "formula_tiebreak",
        },
        "profile_manifest": json.loads(profile_manifest.read_text()),
        "profile_manifest_sha256": file_sha256(profile_manifest),
        "formula_results": validations,
        "cross_formula": {
            "absolute_hit_rate_delta": formula_delta,
            "maximum_allowed_delta": 0.005,
            "passed": cross_formula_check,
        },
    }
    atomic_write_json(args.output_root / "aggregate.json", aggregate)
    print(json.dumps(aggregate, sort_keys=True))
    if aggregate["status"] != "complete":
        failed = {
            formula: [name for name, passed in result["checks"].items() if not passed]
            for formula, result in validations.items()
            if not result["passed"]
        }
        raise RuntimeError(
            f"full-NQ heat contract failed: formulas={failed}, "
            f"cross_formula={cross_formula_check}"
        )


if __name__ == "__main__":
    main()
