#!/usr/bin/env python3
"""Run the fair, resumable MnemoRAG end-to-end comparison campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .benchmark_point import DATASET_PATHS, validate_physical_cpu_ids
    from .comparison_validation import (
        validate_fair_comparison_group,
        validate_mnemorag_full_optimizations,
        validate_standalone_point,
    )
    from .fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
        select_sample_indices,
        validate_trace_workload,
        write_trace_bundle,
    )
    from .slo_campaign import (
        DATASETS,
        QPS_GRIDS,
        SYSTEMS,
        PointGroup,
        candidate_groups,
        dataset_latency_groups,
        final_slo_groups_by_dataset,
        select_boundary_qps_by_dataset,
    )
except ImportError:
    from benchmark_point import DATASET_PATHS, validate_physical_cpu_ids
    from comparison_validation import (
        validate_fair_comparison_group,
        validate_mnemorag_full_optimizations,
        validate_standalone_point,
    )
    from fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
        select_sample_indices,
        validate_trace_workload,
        write_trace_bundle,
    )
    from slo_campaign import (
        DATASETS,
        QPS_GRIDS,
        SYSTEMS,
        PointGroup,
        candidate_groups,
        dataset_latency_groups,
        final_slo_groups_by_dataset,
        select_boundary_qps_by_dataset,
    )


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HEDRA_ROOT = Path(
    os.environ.get("HEDRARAG_ROOT", PROJECT_ROOT / "third_party/HedraRAG")
).resolve()
DEFAULT_PYTHON = Path(os.environ.get("MNEMORAG_PYTHON", sys.executable)).resolve()
EXPECTED_CPU_IDS = tuple(range(64))
EXPECTED_NUMA_NODES = (0, 1, 2, 3)


def parse_numactl_show(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in output.splitlines():
        if ":" not in raw_line:
            continue
        name, value = raw_line.split(":", 1)
        key = name.strip()
        normalized = value.strip()
        if key in {
            "physcpubind",
            "cpubind",
            "nodebind",
            "membind",
            "interleavemask",
            "preferred",
        }:
            parsed[key] = [int(item) for item in normalized.split() if item.isdigit()]
        else:
            parsed[key] = normalized
    return parsed


def current_numactl_policy() -> dict[str, Any]:
    output = subprocess.check_output(
        ["numactl", "--show"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return parse_numactl_show(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("candidate", "final-slo", "dataset-latency"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--mnemo-profile", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--serving-batch-size", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=12288)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--latency-slo-seconds", type=float, default=10.0)
    parser.add_argument("--dataset-qps", type=float, default=4.0)
    parser.add_argument(
        "--fixed-request-count",
        type=int,
        default=512,
        help="Queries per point; the paper evaluates 512 shared queries.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help=(
            "Number of repeated measurements for final-SLO and dataset-latency phases."
        ),
    )
    parser.add_argument(
        "--resource-policy",
        choices=("strict", "expedite"),
        default="strict",
    )
    parser.add_argument(
        "--stop-after-slo-violation",
        action="store_true",
        help=(
            "For candidate QPS scans, retain each system's first p99 > SLO "
            "point and skip that system at higher QPS values."
        ),
    )
    parser.add_argument(
        "--dataset-nprobe",
        type=int,
        choices=(128, 256, 512),
        help=(
            "Restrict the cross-dataset phase to one nprobe; by default it "
            "covers 128, 256, and 512."
        ),
    )
    parser.add_argument(
        "--dataset-nprobes",
        type=int,
        nargs="+",
        choices=(128, 256, 512),
        help="Run the fixed-QPS dataset phase for the listed nprobe values.",
    )
    parser.add_argument(
        "--dataset-workflow",
        choices=("onepass", "recomp", "irg"),
        help=(
            "Restrict the cross-dataset phase to one workflow; by default it "
            "covers all three workflows."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dataset_row_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def question_sha256(question: str) -> str:
    return hashlib.sha256(str(question).encode("utf-8")).hexdigest()


def load_profile(path: Path, dataset: str) -> tuple[list[str], dict[str, str], dict]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("version") != 1:
        raise ValueError("MnemoRAG profile version must be 1")
    arguments = [str(value) for value in profile.get("extra_args", [])]
    environment = {
        str(name): str(value)
        for name, value in (profile.get("environment") or {}).items()
    }
    dataset_environment = (profile.get("dataset_environment") or {}).get(dataset)
    if not isinstance(dataset_environment, dict):
        raise ValueError(f"MnemoRAG profile has no environment for {dataset}")
    environment.update(
        {str(name): str(value) for name, value in dataset_environment.items()}
    )
    required_environment = (
        "MNEMORAG_DOC_HOTNESS_PRIOR_PATH",
        "MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR",
        "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH",
    )
    missing = [name for name in required_environment if not environment.get(name)]
    if missing:
        raise ValueError(f"MnemoRAG profile is missing environment fields: {missing}")
    if environment["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] != "0":
        raise ValueError(
            "document hotness must evolve causally during online measurement"
        )
    heat_option = "--cpu-kv-min-document-heat-to-store"
    if heat_option not in arguments:
        raise ValueError(f"MnemoRAG profile is missing {heat_option}=1")
    heat_position = arguments.index(heat_option)
    if (
        heat_position + 1 >= len(arguments)
        or float(arguments[heat_position + 1]) != 1.0
    ):
        raise ValueError(f"MnemoRAG profile must use {heat_option}=1")
    return arguments, environment, profile


def load_history_indices(profile: dict, dataset: str) -> tuple[np.ndarray, dict]:
    history = (profile.get("dataset_history") or {}).get(dataset)
    if not isinstance(history, dict):
        raise ValueError(f"MnemoRAG profile has no disjoint history for {dataset}")
    required = (
        "sample_indices_path",
        "sample_indices_sha256",
        "prior_path",
        "prior_sha256",
        "query_base_snapshot_path",
        "query_base_snapshot_sha256",
        "question_hashes_path",
        "question_hashes_sha256",
        "dataset_sha256",
        "retrieval_topk",
    )
    missing = [name for name in required if history.get(name) in (None, "")]
    if missing:
        raise ValueError(f"{dataset} history is missing fields: {missing}")
    if int(history["retrieval_topk"]) != 3:
        raise ValueError("history prior must be built from top-3 retrieval results")
    indices_path = Path(history["sample_indices_path"])
    prior_path = Path(history["prior_path"])
    query_base_snapshot_path = Path(history["query_base_snapshot_path"])
    question_hashes_path = Path(history["question_hashes_path"])
    if file_sha256(indices_path) != history["sample_indices_sha256"]:
        raise ValueError(f"{dataset} history sample digest mismatch")
    if file_sha256(prior_path) != history["prior_sha256"]:
        raise ValueError(f"{dataset} history prior digest mismatch")
    if file_sha256(query_base_snapshot_path) != history["query_base_snapshot_sha256"]:
        raise ValueError(f"{dataset} Query Base snapshot digest mismatch")
    if file_sha256(question_hashes_path) != history["question_hashes_sha256"]:
        raise ValueError(f"{dataset} history question digest mismatch")
    dataset_path = DATASET_PATHS[dataset]
    if file_sha256(dataset_path) != history["dataset_sha256"]:
        raise ValueError(f"{dataset} dataset digest mismatch")
    indices = np.asarray(np.load(indices_path, allow_pickle=False), dtype="<i8")
    if indices.shape != (256,):
        raise ValueError(f"{dataset} history must contain exactly 256 rows")
    if len(set(map(int, indices))) != indices.size:
        raise ValueError(f"{dataset} history indices contain duplicates")
    source_count = dataset_row_count(dataset_path)
    if np.any(indices < 0) or np.any(indices >= source_count):
        raise ValueError(f"{dataset} history indices are outside the dataset")
    history_hashes = np.asarray(
        np.load(question_hashes_path, allow_pickle=False), dtype=np.str_
    )
    if history_hashes.shape != indices.shape:
        raise ValueError(f"{dataset} history question hash shape mismatch")
    rows = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    observed_history_hashes = np.asarray(
        [question_sha256(rows[int(index)]["question"]) for index in indices],
        dtype=np.str_,
    )
    if not np.array_equal(history_hashes, observed_history_hashes):
        raise ValueError(f"{dataset} history questions do not match the dataset")
    history_hash_set = set(map(str, history_hashes))
    excluded_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if question_sha256(row["question"]) in history_hash_set
        ],
        dtype="<i8",
    )
    return excluded_indices, history


def validate_disjoint_history(
    sample_indices: np.ndarray,
    history_indices: np.ndarray,
) -> None:
    overlap = np.intersect1d(sample_indices, history_indices, assume_unique=False)
    if overlap.size:
        preview = ", ".join(map(str, overlap[:8]))
        raise ValueError(
            "measurement workload overlaps the frozen history set: " + preview
        )


def verify_driver_topology(gpu_id: str) -> dict:
    cpu_ids = sorted(os.sched_getaffinity(0))
    if cpu_ids != list(EXPECTED_CPU_IDS):
        raise RuntimeError(f"driver CPU affinity must be 0-63, observed {cpu_ids}")
    topology = validate_physical_cpu_ids(cpu_ids)
    if topology.get("physical_core_count") != 64:
        raise RuntimeError("driver does not own 64 physical cores")
    if topology.get("smt_siblings_selected"):
        raise RuntimeError("driver affinity contains SMT siblings")
    if topology.get("socket_ids") != [0]:
        raise RuntimeError(f"driver must use socket 0, observed {topology}")
    if topology.get("numa_node_ids") != list(EXPECTED_NUMA_NODES):
        raise RuntimeError(f"driver must use NUMA 0-3, observed {topology}")
    if str(gpu_id) != "1":
        raise RuntimeError("formal campaign is fixed to physical GPU 1")
    memory_policy = current_numactl_policy()
    expected_nodes = list(EXPECTED_NUMA_NODES)
    if memory_policy.get("policy") != "interleave":
        raise RuntimeError(
            "driver memory policy must interleave NUMA nodes 0-3; "
            f"observed {memory_policy}"
        )
    for field in ("nodebind", "interleavemask"):
        if memory_policy.get(field) != expected_nodes:
            raise RuntimeError(
                f"driver {field} must be NUMA nodes 0-3; observed {memory_policy}"
            )
    topology["memory_policy"] = memory_policy
    return topology


def load_candidate_summaries(path: Path) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("candidate results must be a JSON list")
    return value


def build_plan(args: argparse.Namespace) -> list[PointGroup]:
    fixed_count = getattr(args, "fixed_request_count", None)
    if fixed_count is not None and int(fixed_count) <= 0:
        raise ValueError("--fixed-request-count must be positive")
    repeats = int(getattr(args, "repeats", 2))
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    request_bounds = (
        {
            "minimum_requests": int(fixed_count),
            "maximum_requests": int(fixed_count),
        }
        if fixed_count is not None
        else {}
    )
    if args.phase == "candidate":
        return candidate_groups(
            datasets=DATASETS,
            nprobes=(128,),
            **request_bounds,
        )
    if args.phase == "final-slo":
        if args.candidate_results is None:
            raise ValueError("--candidate-results is required for final-slo")
        selected = select_boundary_qps_by_dataset(
            load_candidate_summaries(args.candidate_results),
            latency_slo_seconds=args.latency_slo_seconds,
            nprobes=(128,),
        )
        return final_slo_groups_by_dataset(
            selected,
            nprobes=(128,),
            repeats=repeats,
            **request_bounds,
        )
    if args.dataset_nprobe is not None and args.dataset_nprobes is not None:
        raise ValueError("use either --dataset-nprobe or --dataset-nprobes")
    return dataset_latency_groups(
        qps=args.dataset_qps,
        nprobe=args.dataset_nprobe,
        nprobes=args.dataset_nprobes,
        workflow=args.dataset_workflow,
        repeats=repeats,
        num_requests=(int(fixed_count) if fixed_count is not None else 1024),
    )


def common_arguments(
    args: argparse.Namespace,
    group: PointGroup,
    record: dict,
    trace_path: Path,
    output_dir: Path,
) -> list[str]:
    expedited = getattr(args, "resource_policy", "strict") == "expedite"
    common = [
        "--dataset",
        group.dataset,
        "--workflow",
        group.workflow,
        "--num-requests",
        str(group.num_requests),
        "--arrival-rate",
        f"{group.qps:g}",
        "--arrival-burstiness",
        "1",
        "--normalize-arrival-trace",
        "--arrival-trace",
        str(trace_path),
        "--latency-slo-seconds",
        f"{group.latency_slo_seconds:g}",
        "--campaign-group-id",
        group.identifier(),
        "--campaign-repeat",
        str(group.repeat),
        "--campaign-order-position",
        str(record["order_position"]),
        "--seed",
        str(group.seed),
        "--topk",
        str(group.topk),
        "--nprobe",
        str(group.nprobe),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--gpu-id",
        args.gpu_id,
        "--cpu-threads",
        "64",
        "--require-clean-resources",
        "--max-preflight-cpu-busy-percent",
        "100" if expedited else "5",
        "--max-preflight-cpu-max-busy-percent",
        "100" if expedited else "5",
        "--max-preflight-gpu-memory-mib",
        "256",
        "--max-preflight-gpu-util-percent",
        "5",
        "--runtime-resource-sample-seconds",
        "2",
        "--max-runtime-external-cpu-busy-percent",
        "100" if expedited else "5",
        "--output-dir",
        str(output_dir),
    ]
    return common


def build_system_command(
    args: argparse.Namespace,
    group: PointGroup,
    record: dict,
    trace_path: Path,
    output_dir: Path,
    mnemo_extra_args: list[str],
) -> tuple[list[str], Path]:
    system = record["system"]
    prefix = [
        "numactl",
        "--physcpubind=0-63",
        "--interleave=0-3",
        str(args.python),
    ]
    common = common_arguments(
        args,
        group,
        record,
        trace_path,
        output_dir,
    )
    if system == "mnemorag":
        workflow_specific = (
            [
                "--cache-aware-prefill-batching",
                "--no-cross-microbatch-pipeline",
            ]
            if group.workflow == "onepass"
            else [
                "--no-cache-aware-prefill-batching",
                "--cross-microbatch-pipeline",
            ]
        )
        command = (
            prefix
            + [str(HERE / "benchmark_point.py")]
            + common
            + [
                "--components",
                "full",
                "--order-mode",
                "hotness_hint",
                "--retrieval-batch-size",
                str(args.retrieval_batch_size),
                "--microbatch-size",
                str(args.serving_batch_size),
                "--worker-count",
                "8",
                "--async-retrieval-policy",
                "prefill_batch",
                "--no-direct-vllm-dispatch",
                "--async-generation-queue-batches",
                "2",
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--faiss-omp-threads",
                "1",
                "--faiss-search-workers",
                "64",
                "--drop-loaded-input-page-cache",
                "--prefix-caching",
                "--record-prefix-distribution",
                "--record-stage-breakdown",
            ]
            + workflow_specific
            + mnemo_extra_args
        )
        return command, PROJECT_ROOT
    command = (
        prefix
        + [str(HERE / "hedra_flash_point.py")]
        + common
        + [
            "--system",
            system,
            "--retrieval-batch-size",
            str(args.retrieval_batch_size),
            "--serving-batch-size",
            str(args.serving_batch_size),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--recomp-compression-max-tokens",
            "128",
        ]
    )
    return command, HEDRA_ROOT


def prepare_trace(
    group: PointGroup,
    group_dir: Path,
    *,
    excluded_indices,
) -> Path:
    trace_path = group_dir / "workload.npz"
    dataset_path = DATASET_PATHS[group.dataset]
    dataset_digest = file_sha256(dataset_path)
    spec = ArrivalTraceSpec(
        request_count=group.num_requests,
        request_rate=group.qps,
        burstiness=1.0,
        random_seed=group.seed,
        normalize_duration=True,
    )
    if trace_path.exists():
        offsets, manifest = load_trace_bundle(trace_path)
        indices = load_trace_sample_indices(trace_path)
        validate_trace_workload(
            manifest,
            spec,
            sample_indices=indices,
            dataset_sha256=dataset_digest,
        )
        validate_disjoint_history(indices, np.asarray(excluded_indices, dtype="<i8"))
        return trace_path
    source_count = dataset_row_count(dataset_path)
    indices = select_sample_indices(
        source_count,
        group.num_requests,
        random_seed=group.seed,
        excluded_indices=excluded_indices,
    )
    validate_disjoint_history(indices, np.asarray(excluded_indices, dtype="<i8"))
    offsets = build_arrival_offsets(spec)
    write_trace_bundle(
        trace_path,
        spec,
        offsets,
        sample_indices=indices,
        dataset_sha256=dataset_digest,
    )
    return trace_path


def find_summary(output_dir: Path) -> Path:
    paths = list(output_dir.glob("*/summary.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one summary under {output_dir}, observed {paths}")
    return paths[0]


def load_accepted_group(
    group_dir: Path,
    history_metadata: dict,
) -> list[dict] | None:
    marker = group_dir / "accepted.json"
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("disjoint_history") != history_metadata:
        raise ValueError(
            f"accepted group {group_dir} was produced with different history data"
        )
    summaries = [
        json.loads((group_dir / relative).read_text(encoding="utf-8"))
        for relative in metadata["summary_paths"]
    ]
    if metadata.get("evaluation_scope") == "slo_censored_partial":
        for summary in summaries:
            validate_standalone_point(summary)
        mnemo = next(
            (summary for summary in summaries if summary["system"] == "mnemorag"),
            None,
        )
        if mnemo is not None:
            validate_mnemorag_full_optimizations(mnemo)
        return summaries
    validate_fair_comparison_group(
        summaries,
        required_systems=tuple(metadata["group"]["system_order"]),
    )
    validate_mnemorag_full_optimizations(
        next(summary for summary in summaries if summary["system"] == "mnemorag")
    )
    return summaries


def run_group(
    args: argparse.Namespace,
    group: PointGroup,
    mnemo_extra_args: list[str],
    mnemo_environment: dict[str, str],
    history_indices,
    history_metadata: dict,
    skip_systems: set[str] | None = None,
) -> list[dict]:
    group_dir = args.output_dir / "groups" / group.identifier()
    accepted = load_accepted_group(group_dir, history_metadata)
    if accepted is not None:
        print(f"SKIP_ACCEPTED {group.identifier()}", flush=True)
        return accepted
    group_dir.mkdir(parents=True, exist_ok=True)
    attempts = sorted(group_dir.glob("attempt_*"))
    attempt_dir = group_dir / f"attempt_{len(attempts) + 1:03d}"
    attempt_dir.mkdir(parents=True)
    trace_path = prepare_trace(
        group,
        group_dir,
        excluded_indices=history_indices,
    )
    summaries = []
    summary_paths = []
    skipped = set(skip_systems or set())
    for record in group.records():
        system = record["system"]
        if system in skipped:
            continue
        output_dir = attempt_dir / system
        command, cwd = build_system_command(
            args,
            group,
            record,
            trace_path,
            output_dir,
            mnemo_extra_args,
        )
        environment = os.environ.copy()
        if system == "mnemorag":
            environment.update(mnemo_environment)
        command_log = output_dir / "command.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            command_log,
            {
                "argv": command,
                "cwd": str(cwd),
                "environment_overrides": (
                    mnemo_environment if system == "mnemorag" else {}
                ),
            },
        )
        print(
            f"RUN {group.identifier()} {record['order_position'] + 1}/"
            f"{len(group.system_order)} {system}",
            flush=True,
        )
        with (output_dir / "driver.log").open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
        summary_path = find_summary(output_dir)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        summary_paths.append(str(summary_path.relative_to(group_dir)))

    mnemo_summary = next(
        (summary for summary in summaries if summary["system"] == "mnemorag"),
        None,
    )
    mnemo_validation = (
        validate_mnemorag_full_optimizations(mnemo_summary)
        if mnemo_summary is not None
        else {
            "status": "not_run",
            "reason": "system already crossed the latency SLO",
        }
    )
    if len(summaries) == len(group.system_order):
        validation = validate_fair_comparison_group(
            summaries,
            required_systems=group.system_order,
        )
    else:
        validation = {
            "status": "passed",
            "mode": "slo_censored_partial",
            "systems_measured": [summary["system"] for summary in summaries],
            "systems_skipped": sorted(skipped),
            "standalone_validations": [
                validate_standalone_point(summary)
                for summary in summaries
            ],
        }
    atomic_write_json(
        group_dir / "accepted.json",
        {
            "status": "accepted",
            "accepted_at_unix_s": time.time(),
            "group": asdict(group),
            "summary_paths": summary_paths,
            "comparison_validation": validation,
            "mnemo_full_optimization_validation": mnemo_validation,
            "disjoint_history": history_metadata,
            "evaluation_scope": (
                "matched_fair_subset"
                if len(summaries) == len(group.system_order)
                else "slo_censored_partial"
            ),
            "slo_censored_systems": sorted(skipped),
            "resource_policy": {
                "mode": getattr(args, "resource_policy", "strict"),
                "cpu_busy_threshold_percent": (
                    100
                    if getattr(args, "resource_policy", "strict") == "expedite"
                    else 5
                ),
                "gpu_exclusivity_required": True,
                "resource_monitoring_enabled": True,
            },
        },
    )
    return summaries


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / f"{args.phase}_plan.json",
        [asdict(group) for group in plan],
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "group_count": len(plan),
                    "run_count": sum(len(group.system_order) for group in plan),
                },
                sort_keys=True,
            )
        )
        return
    if args.mnemo_profile is None:
        raise ValueError("--mnemo-profile is required for an executable campaign")
    verify_driver_topology(args.gpu_id)
    all_summaries = []
    crossed_slo: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    censoring_records: list[dict[str, Any]] = []
    planned_scan_keys = sorted(
        {(group.dataset, group.workflow, int(group.nprobe)) for group in plan}
    )

    def candidate_censoring_payload(*, complete: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": "per-system stop after first p99 > latency SLO",
            "latency_slo_seconds": 10.0,
            "qps_cap_by_workflow": {
                workflow: max(map(float, grid)) for workflow, grid in QPS_GRIDS.items()
            },
            "extrapolation_beyond_qps_cap": False,
            "crossings": {
                "|".join(map(str, key)): value for key, value in crossed_slo.items()
            },
            "skipped_groups": censoring_records,
        }
        if complete:
            payload["systems_reaching_qps_cap_without_violation"] = [
                {
                    "dataset": dataset,
                    "workflow": workflow,
                    "nprobe": nprobe,
                    "system": system,
                    "qps_cap": max(map(float, QPS_GRIDS[workflow])),
                    "capacity_interpretation": "right_censored_lower_bound",
                }
                for dataset, workflow, nprobe in planned_scan_keys
                for system in SYSTEMS
                if system not in crossed_slo.get((dataset, workflow, nprobe), {})
            ]
        return payload

    frozen_profile = None
    dataset_profile_cache = {}
    for index, group in enumerate(plan, 1):
        scan_key = (group.dataset, group.workflow, int(group.nprobe))
        skipped_systems = (
            set(crossed_slo.get(scan_key, {}))
            if args.phase == "candidate" and args.stop_after_slo_violation
            else set()
        )
        if skipped_systems == set(group.system_order):
            censoring_records.append(
                {
                    "group": group.identifier(),
                    "qps": group.qps,
                    "systems_skipped": sorted(skipped_systems),
                    "reason": "all systems previously crossed p99 latency SLO",
                }
            )
            atomic_write_json(
                args.output_dir / "candidate_slo_censoring.json",
                candidate_censoring_payload(),
            )
            print(f"SKIP_SLO_CENSORED_GROUP {group.identifier()}", flush=True)
            continue
        if group.dataset not in dataset_profile_cache:
            extra_args, environment, profile = load_profile(
                args.mnemo_profile, group.dataset
            )
            history_indices, history_metadata = load_history_indices(
                profile, group.dataset
            )
            dataset_profile_cache[group.dataset] = (
                extra_args,
                environment,
                profile,
                history_indices,
                history_metadata,
            )
        (
            extra_args,
            environment,
            profile,
            history_indices,
            history_metadata,
        ) = dataset_profile_cache[group.dataset]
        group_environment = dict(environment)
        prior_from_environment = Path(
            environment["MNEMORAG_DOC_HOTNESS_PRIOR_PATH"]
        ).resolve()
        prior_from_history = Path(history_metadata["prior_path"]).resolve()
        if prior_from_environment != prior_from_history:
            raise ValueError(
                f"{group.dataset} prior path differs between environment and history"
            )
        snapshot_from_environment = Path(
            environment["MNEMORAG_QUERY_BASE_SNAPSHOT_PATH"]
        ).resolve()
        snapshot_from_history = Path(
            history_metadata["query_base_snapshot_path"]
        ).resolve()
        if snapshot_from_environment != snapshot_from_history:
            raise ValueError(
                f"{group.dataset} Query Base snapshot path differs between "
                "environment and history"
            )
        if frozen_profile is None:
            frozen_profile = profile
            atomic_write_json(args.output_dir / "mnemo_profile.json", profile)
        print(f"GROUP {index}/{len(plan)} {group.identifier()}", flush=True)
        group_summaries = run_group(
            args,
            group,
            extra_args,
            group_environment,
            history_indices,
            history_metadata,
            skip_systems=skipped_systems,
        )
        all_summaries.extend(group_summaries)
        if args.phase == "candidate" and args.stop_after_slo_violation:
            for summary in group_summaries:
                p99 = float(summary["request_latency_p99_s"])
                if p99 > float(group.latency_slo_seconds):
                    crossed_slo.setdefault(scan_key, {}).setdefault(
                        summary["system"],
                        {
                            "group": group.identifier(),
                            "qps": group.qps,
                            "p99_latency_s": p99,
                            "latency_slo_seconds": group.latency_slo_seconds,
                        },
                    )
            atomic_write_json(
                args.output_dir / "candidate_slo_censoring.json",
                candidate_censoring_payload(),
            )
        atomic_write_json(
            args.output_dir / f"{args.phase}_results.json",
            all_summaries,
        )
    if args.phase == "candidate" and args.stop_after_slo_violation:
        atomic_write_json(
            args.output_dir / "candidate_slo_censoring.json",
            candidate_censoring_payload(complete=True),
        )
    print(
        f"FULL_E2E_PHASE_COMPLETE {args.phase} groups={len(plan)} ",
        f"runs={len(all_summaries)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
