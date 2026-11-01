#!/usr/bin/env python3
"""Run the fair, resumable MnemoRAG end-to-end comparison campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
        mnemo_full_groups,
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
        mnemo_full_groups,
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
        choices=("candidate", "final-slo", "dataset-latency", "mnemo-full"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument(
        "--reuse-campaign-root",
        type=Path,
        help=(
            "Read accepted baseline points from an older campaign. Reused rows "
            "are copied into this campaign with explicit legacy_semantics provenance."
        ),
    )
    parser.add_argument(
        "--reuse-mnemo-campaign-root",
        type=Path,
        help=(
            "Reuse only hash- and command-identical MnemoRAG candidate points "
            "from an older campaign. FlashRAG and HedraRAG are always rerun."
        ),
    )
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
        help="Use the same request count for every planned point.",
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
        "--selection-root",
        type=Path,
        help=(
            "Directory containing one audited sample_indices.npy and "
            "selection_manifest.json per dataset."
        ),
    )
    parser.add_argument(
        "--next-use-profile-root",
        type=Path,
        help=(
            "Root containing dataset/workflow/nprobeN/next_use.json assets "
            "for profile-guided online serving."
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
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{dataset} history indices must be a nonempty vector")
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
    if args.phase == "mnemo-full":
        if args.candidate_results is None:
            raise ValueError("--candidate-results is required for mnemo-full")
        if args.mnemo_profile is None:
            raise ValueError("--mnemo-profile is required for mnemo-full")
        selected = {
            (dataset, workflow, 128): tuple(map(float, QPS_GRIDS[workflow]))
            for dataset in DATASETS
            for workflow in ("onepass", "recomp", "irg")
        }
        selected.update(
            {
                (dataset, workflow, nprobe): (float(args.dataset_qps),)
                for dataset in DATASETS
                for workflow in ("onepass", "recomp", "irg")
                for nprobe in (256, 512)
            }
        )
        request_counts = {}
        for dataset in DATASETS:
            _, _, profile = load_profile(args.mnemo_profile, dataset)
            history_indices, _ = load_history_indices(profile, dataset)
            request_counts[dataset] = dataset_row_count(DATASET_PATHS[dataset]) - len(
                set(map(int, history_indices))
            )
        return mnemo_full_groups(selected, request_counts)
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
    selection_evidence: dict[str, Any] | None = None,
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
    warmup_path = (selection_evidence or {}).get("warmup_indices_path")
    if warmup_path is not None:
        common += ["--online-warmup-indices", str(warmup_path)]
    return common


def audited_hedra_hot_cluster_csv(
    group: PointGroup,
    selection_evidence: dict[str, Any] | None,
) -> Path:
    if selection_evidence is None:
        raise ValueError(
            "HedraRAG formal measurements require audited selection evidence"
        )
    dataset_dir = Path(selection_evidence["selection_dataset_dir"]).resolve()
    profile_path = dataset_dir / "hedrarag_hot_cluster_profile.json"
    if not profile_path.is_file():
        raise ValueError(f"missing HedraRAG hot-cluster profile: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("dataset_sha256") != selection_evidence.get("dataset_sha256")
        or profile.get("sample_indices_sha256")
        != selection_evidence.get("sample_indices_sha256")
        or profile.get("warmup_indices_sha256")
        != selection_evidence.get("warmup_indices_sha256")
        or int(profile.get("warmup_measurement_overlap", -1)) != 0
    ):
        raise ValueError(f"invalid HedraRAG hot-cluster profile: {profile_path}")
    entry = (profile.get("nprobes") or {}).get(str(group.nprobe)) or {}
    csv_path = Path(str(entry.get("csv_path", ""))).resolve()
    expected_path = dataset_dir / f"hedrarag_hot_clusters_np{group.nprobe}.csv"
    hot_policy = entry.get("hot_policy") or {}
    fallback = entry.get("largest_cluster_fallback") or {}
    if (
        csv_path != expected_path.resolve()
        or not csv_path.is_file()
        or entry.get("csv_sha256") != file_sha256(csv_path)
        or int(entry.get("nprobe", -1)) != int(group.nprobe)
        or entry.get("ordering_source") != "disjoint_warmup_cluster_frequency"
        or float(hot_policy.get("cluster_hit_rate", 0.0))
        <= float(fallback.get("cluster_hit_rate", 0.0))
    ):
        raise ValueError(f"HedraRAG hot-cluster CSV failed validation: {csv_path}")
    return csv_path


def build_system_command(
    args: argparse.Namespace,
    group: PointGroup,
    record: dict,
    trace_path: Path,
    output_dir: Path,
    mnemo_extra_args: list[str],
    selection_evidence: dict[str, Any] | None = None,
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
        selection_evidence,
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
                "--no-cross-microbatch-pipeline",
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
                "eager",
                "--direct-vllm-dispatch",
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
    if system == "hedrarag":
        command += [
            "--hot-cluster-csv",
            str(audited_hedra_hot_cluster_csv(group, selection_evidence)),
        ]
    return command, HEDRA_ROOT


def next_use_profile_path(root: Path, group: PointGroup) -> Path:
    return (
        Path(root)
        / group.dataset
        / group.workflow
        / f"nprobe{group.nprobe}"
        / "next_use.json"
    ).resolve()


def load_preselected_indices(
    selection_root: Path,
    dataset: str,
    *,
    request_count: int,
    dataset_sha256: str,
    excluded_indices,
) -> tuple[np.ndarray, dict[str, Any]]:
    dataset_dir = Path(selection_root) / dataset
    manifest_path = dataset_dir / "selection_manifest.json"
    indices_path = dataset_dir / "sample_indices.npy"
    warmup_indices_path = dataset_dir / "warmup_indices.npy"
    if (
        not manifest_path.is_file()
        or not indices_path.is_file()
        or not warmup_indices_path.is_file()
    ):
        raise ValueError(f"missing audited selection for {dataset}: {dataset_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mnemorag.prefix-equivalent-subset.v4":
        raise ValueError(f"unsupported selection format for {dataset}")
    if manifest.get("dataset") != dataset:
        raise ValueError(f"selection dataset mismatch for {dataset}")
    if int(manifest.get("selected_rows", -1)) != int(request_count):
        raise ValueError(
            f"selection for {dataset} has {manifest.get('selected_rows')} rows; "
            f"expected {request_count}"
        )
    artifacts = manifest.get("artifacts") or {}
    if artifacts.get("dataset_sha256") != dataset_sha256:
        raise ValueError(f"selection dataset digest mismatch for {dataset}")
    if artifacts.get("sample_indices_sha256") != file_sha256(indices_path):
        raise ValueError(f"selection index digest mismatch for {dataset}")
    if artifacts.get("warmup_indices_sha256") != file_sha256(warmup_indices_path):
        raise ValueError(f"selection warmup digest mismatch for {dataset}")
    validation = manifest.get("validation") or {}
    required_validations = (
        "offline_acceptance",
        "unique_selected_rows",
        "prefix_distribution_within_tolerance",
        "online_action_mix_within_tolerance",
        "hot_request_fraction_within_tolerance",
        "h2d_proxy_within_tolerance",
        "reuse_distance_distribution_within_tolerance",
        "unique_warmup_rows",
    )
    if (
        any(validation.get(key) is not True for key in required_validations)
        or int(validation.get("frozen_history_overlap_rows", -1)) != 0
        or int(validation.get("warmup_frozen_history_overlap_rows", -1)) != 0
        or int(validation.get("warmup_measurement_overlap_rows", -1)) != 0
    ):
        raise ValueError(f"selection validation failed for {dataset}")
    prefix_contract = manifest.get("prefix_contract") or {}
    if (
        int(prefix_contract.get("topk", -1)) != 3
        or prefix_contract.get("document_hotness_freeze_prior") is not False
        or int(prefix_contract.get("cpu_kv_min_document_heat_to_store", -1)) != 1
        or prefix_contract.get("measurement_excludes_warmup") is not True
        or prefix_contract.get("warmup_and_measurement_disjoint") is not True
        or prefix_contract.get("runtime_validation_required") is not True
    ):
        raise ValueError(f"selection prefix contract failed for {dataset}")
    indices = np.asarray(np.load(indices_path, allow_pickle=False), dtype="<i8")
    if indices.shape != (int(request_count),):
        raise ValueError(f"selection index shape mismatch for {dataset}")
    if len(set(map(int, indices))) != len(indices):
        raise ValueError(f"selection contains duplicate rows for {dataset}")
    warmup_indices = np.asarray(
        np.load(warmup_indices_path, allow_pickle=False), dtype="<i8"
    )
    if warmup_indices.shape != (int(request_count),):
        raise ValueError(f"selection warmup shape mismatch for {dataset}")
    if len(set(map(int, warmup_indices))) != len(warmup_indices):
        raise ValueError(f"selection warmup contains duplicate rows for {dataset}")
    if np.intersect1d(indices, warmup_indices).size:
        raise ValueError(f"selection warmup overlaps measurement for {dataset}")
    validate_disjoint_history(
        indices,
        np.asarray(excluded_indices, dtype="<i8"),
    )
    validate_disjoint_history(
        warmup_indices,
        np.asarray(excluded_indices, dtype="<i8"),
    )
    return indices, {
        "selection_dataset_dir": str(dataset_dir.resolve()),
        "dataset_sha256": dataset_sha256,
        "selection_manifest_path": str(manifest_path.resolve()),
        "selection_manifest_sha256": file_sha256(manifest_path),
        "sample_indices_path": str(indices_path.resolve()),
        "sample_indices_sha256": file_sha256(indices_path),
        "warmup_indices_path": str(warmup_indices_path.resolve()),
        "warmup_indices_sha256": file_sha256(warmup_indices_path),
        "selection_semantics": manifest.get("selection_semantics"),
        "selection_format": manifest.get("format"),
        "prefix_contract": prefix_contract,
        "validation": validation,
        "headline_disclosure_required": bool(
            manifest.get("headline_disclosure_required")
        ),
    }


def prepare_trace(
    group: PointGroup,
    group_dir: Path,
    *,
    excluded_indices,
    selection_root: Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
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
        selection_evidence = None
        if selection_root is not None:
            expected, selection_evidence = load_preselected_indices(
                selection_root,
                group.dataset,
                request_count=group.num_requests,
                dataset_sha256=dataset_digest,
                excluded_indices=excluded_indices,
            )
            if not np.array_equal(indices, expected):
                raise ValueError(
                    f"existing trace does not use the audited {group.dataset} selection"
                )
        return trace_path, selection_evidence
    source_count = dataset_row_count(dataset_path)
    selection_evidence = None
    if selection_root is None:
        indices = select_sample_indices(
            source_count,
            group.num_requests,
            random_seed=group.seed,
            excluded_indices=excluded_indices,
        )
    else:
        indices, selection_evidence = load_preselected_indices(
            selection_root,
            group.dataset,
            request_count=group.num_requests,
            dataset_sha256=dataset_digest,
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
    return trace_path, selection_evidence


def legacy_group_sources(
    reuse_root: Path | None,
    group: PointGroup,
) -> tuple[Path | None, dict[str, tuple[dict, Path, Path]]]:
    if reuse_root is None:
        return None, {}
    source_group = Path(reuse_root) / "groups" / group.identifier()
    marker = source_group / "accepted.json"
    if not marker.exists():
        return None, {}
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    sources: dict[str, tuple[dict, Path, Path]] = {}
    for relative in metadata.get("summary_paths", []):
        summary_path = source_group / relative
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        system = str(summary.get("system"))
        command_path = summary_path.parent.parent / "command.json"
        if system not in {"flashrag", "hedrarag"} or not command_path.exists():
            continue
        command = json.loads(command_path.read_text(encoding="utf-8"))
        argv = command.get("argv") or []
        if argv[:3] != [
            "numactl",
            "--physcpubind=0-63",
            "--interleave=0-3",
        ]:
            raise ValueError(
                f"legacy command does not prove fixed topology: {command_path}"
            )
        sources[system] = (summary, summary_path, command_path)
    return source_group, sources


def materialize_legacy_baseline(
    group_dir: Path,
    source: tuple[dict, Path, Path],
) -> tuple[dict, str]:
    source_summary, source_summary_path, source_command_path = source
    system = str(source_summary["system"])
    summary = dict(source_summary)
    summary["numa_policy"] = summary.get("numa_policy") or {
        "policy": "interleave",
        "physcpubind": " ".join(map(str, range(64))),
        "cpubind": "0 1 2 3",
        "nodebind": "0 1 2 3",
        "membind": "0 1 2 3 4 5 6 7",
        "interleavemask": "0 1 2 3",
    }
    summary["provenance"] = {
        "classification": "legacy_semantics",
        "reused_without_remeasurement": True,
        "source_summary_path": str(source_summary_path.resolve()),
        "source_summary_sha256": file_sha256(source_summary_path),
        "source_command_path": str(source_command_path.resolve()),
        "source_command_sha256": file_sha256(source_command_path),
        "normalization": (
            "numa_policy reconstructed only when the immutable command manifest "
            "proves --physcpubind=0-63 and --interleave=0-3"
        ),
    }
    summary["headline_eligible"] = system != "hedrarag"
    destination = group_dir / "reused_baselines" / system / "summary.json"
    atomic_write_json(destination, summary)
    return summary, str(destination.relative_to(group_dir))


def _source_group_dir(reuse_root: Path, group: PointGroup) -> Path:
    root = Path(reuse_root).resolve()
    direct = root / "groups" / group.identifier()
    if direct.is_dir():
        return direct
    return root / "candidate" / "groups" / group.identifier()


def _validate_mnemo_source_files(reuse_root: Path) -> dict[str, Any]:
    root = Path(reuse_root).resolve()
    provenance_path = next(
        (
            path
            for path in (
                root / "source_provenance_start.json",
                root.parent / "source_provenance_start.json",
            )
            if path.is_file()
        ),
        None,
    )
    if provenance_path is None:
        raise ValueError(f"missing source provenance for MnemoRAG reuse: {root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    repositories = provenance.get("repositories") or {}
    required = {
        "mnemorag_flashrag": {
            "benchmarks/benchmark_point.py",
            "benchmarks/comparison_validation.py",
            "benchmarks/fair_workload.py",
            "mnemorag/generator/generator.py",
            "mnemorag/retriever/retriever.py",
        },
    }
    verified: dict[str, str] = {}
    for repository_name, repository in repositories.items():
        entries = {
            str(entry["path"]): str(entry["sha256"])
            for entry in repository.get("files") or []
        }
        paths = set(required.get(repository_name, set()))
        if repository_name == "patched_vllm":
            paths.update(path for path in entries if "mnemorag" in path.lower())
        for relative in sorted(paths):
            expected = entries.get(relative)
            source_path = Path(repository["root"]) / relative
            if expected is None or not source_path.is_file():
                raise ValueError(
                    f"missing source evidence for reusable MnemoRAG file: {source_path}"
                )
            observed = file_sha256(source_path)
            if observed != expected:
                raise ValueError(
                    f"MnemoRAG source changed since reused measurement: {source_path}"
                )
            verified[str(source_path.resolve())] = observed
    return {
        "source_provenance_path": str(provenance_path.resolve()),
        "source_provenance_sha256": file_sha256(provenance_path),
        "verified_file_count": len(verified),
        "verified_files_sha256": hashlib.sha256(
            json.dumps(verified, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _normalized_reuse_command(argv: list[str]) -> list[str]:
    normalized = list(map(str, argv))
    for option in ("--arrival-trace", "--output-dir"):
        if option in normalized:
            index = normalized.index(option)
            normalized[index + 1] = f"<{option[2:]}>"
    return normalized


def reusable_mnemo_source(
    reuse_root: Path | None,
    group: PointGroup,
    trace_path: Path,
    expected_command: list[str],
    selection_evidence: dict[str, Any] | None,
) -> tuple[dict, Path, Path, dict[str, Any]] | None:
    if reuse_root is None:
        return None
    source_group = _source_group_dir(reuse_root, group)
    marker = source_group / "accepted.json"
    if not marker.is_file():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    source_summary_path = None
    source_summary = None
    for relative in metadata.get("summary_paths") or []:
        candidate = source_group / relative
        summary = json.loads(candidate.read_text(encoding="utf-8"))
        if summary.get("system") == "mnemorag":
            source_summary_path = candidate
            source_summary = summary
            break
    if source_summary_path is None or source_summary is None:
        return None
    source_command_path = source_summary_path.parent.parent / "command.json"
    if not source_command_path.is_file():
        raise ValueError(f"missing reusable MnemoRAG command: {source_command_path}")
    source_command = json.loads(source_command_path.read_text(encoding="utf-8"))
    if _normalized_reuse_command(source_command.get("argv") or []) != (
        _normalized_reuse_command(expected_command)
    ):
        raise ValueError(
            f"MnemoRAG command differs from reusable point: {source_command_path}"
        )

    _, trace_manifest = load_trace_bundle(trace_path)
    trace_indices = list(map(int, load_trace_sample_indices(trace_path)))
    expected_fields = {
        "system": "mnemorag",
        "dataset": group.dataset,
        "workflow": group.workflow,
        "topk": group.topk,
        "nprobe": group.nprobe,
        "num_requests": group.num_requests,
        "seed": group.seed,
        "campaign_group_id": group.identifier(),
    }
    for field, expected in expected_fields.items():
        if source_summary.get(field) != expected:
            raise ValueError(
                f"reusable MnemoRAG {field} mismatch: "
                f"{source_summary.get(field)!r} != {expected!r}"
            )
    if float(source_summary.get("arrival_rate_rps", -1)) != float(group.qps):
        raise ValueError("reusable MnemoRAG QPS mismatch")
    if (
        source_summary.get("arrival_trace_sha256")
        != trace_manifest["arrival_trace_sha256"]
        or list(map(int, source_summary.get("sample_indices") or [])) != trace_indices
    ):
        raise ValueError("reusable MnemoRAG workload differs from current trace")
    warmup = source_summary.get("online_warmup") or {}
    if selection_evidence is not None and (
        warmup.get("indices_sha256") != selection_evidence.get("warmup_indices_sha256")
    ):
        raise ValueError("reusable MnemoRAG online warmup differs")
    artifact = source_summary.get("request_metrics_artifact") or {}
    artifact_path = Path(str(artifact.get("path", "")))
    if (
        not artifact_path.is_file()
        or artifact.get("sha256") != file_sha256(artifact_path)
        or int(artifact.get("request_count", -1)) != group.num_requests
    ):
        raise ValueError("reusable MnemoRAG request metrics are invalid")
    source_validation = _validate_mnemo_source_files(Path(reuse_root))
    return (
        source_summary,
        source_summary_path,
        source_command_path,
        source_validation,
    )


def materialize_reused_mnemo(
    group_dir: Path,
    source: tuple[dict, Path, Path, dict[str, Any]],
) -> tuple[dict, str]:
    source_summary, source_summary_path, source_command_path, validation = source
    destination_dir = group_dir / "reused_mnemo"
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_artifact = Path(source_summary["request_metrics_artifact"]["path"])
    destination_artifact = destination_dir / "request_metrics.npz"
    shutil.copy2(source_artifact, destination_artifact)
    summary = dict(source_summary)
    summary["request_metrics_artifact"] = {
        **source_summary["request_metrics_artifact"],
        "path": str(destination_artifact.resolve()),
        "sha256": file_sha256(destination_artifact),
    }
    summary["provenance"] = {
        "classification": "hash_verified_current_mnemo_semantics",
        "reused_without_remeasurement": True,
        "source_summary_path": str(source_summary_path.resolve()),
        "source_summary_sha256": file_sha256(source_summary_path),
        "source_command_path": str(source_command_path.resolve()),
        "source_command_sha256": file_sha256(source_command_path),
        **validation,
    }
    summary["headline_eligible"] = True
    destination_summary = destination_dir / "summary.json"
    atomic_write_json(destination_summary, summary)
    shutil.copy2(source_command_path, destination_dir / "source_command.json")
    return summary, str(destination_summary.relative_to(group_dir))


def find_summary(output_dir: Path) -> Path:
    paths = list(output_dir.glob("*/summary.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one summary under {output_dir}, observed {paths}")
    return paths[0]


def validate_profile_guided_summary(
    summary: dict[str, Any], expected_profile_path: Path
) -> dict[str, Any]:
    metadata = summary.get("profile_guided_online_serving") or {}
    stats = summary.get("kv_cache_stats_total_including_warmup") or {}
    accesses = int(stats.get("profile_next_use_accesses", 0) or 0)
    matches = int(stats.get("profile_next_use_matches", 0) or 0)
    finite = int(stats.get("profile_next_use_finite", 0) or 0)
    match_rate = matches / accesses if accesses else 0.0
    expected_profile_path = expected_profile_path.resolve()
    checks = {
        "metadata_enabled": metadata.get("enabled") is True,
        "profile_path_matches": (
            Path(str(metadata.get("path", ""))).resolve() == expected_profile_path
        ),
        "profile_sha256_matches": (
            metadata.get("sha256") == file_sha256(expected_profile_path)
        ),
        "offline_build_excluded": (
            metadata.get("offline_profile_build_excluded_from_latency") is True
        ),
        "profile_is_observation_only": metadata.get("observe_only") is True,
        "profile_does_not_influence_eviction": (
            metadata.get("influences_eviction") is False
        ),
        "profile_accesses_nonzero": accesses > 0,
        "profile_match_rate_at_least_99_percent": match_rate >= 0.99,
        "finite_future_reuse_nonzero": finite > 0,
    }
    if not all(checks.values()):
        raise ValueError(
            "profile-guided serving validation failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "passed",
        "checks": checks,
        "accesses": accesses,
        "matches": matches,
        "match_rate": match_rate,
        "finite_future_reuse_accesses": finite,
        "profile_path": str(expected_profile_path),
        "profile_sha256": file_sha256(expected_profile_path),
    }


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
    if metadata.get("evaluation_scope") == "full_split_long_horizon":
        if len(summaries) != 1 or summaries[0].get("system") != "mnemorag":
            raise ValueError(f"invalid full-split marker in {group_dir}")
        validate_standalone_point(summaries[0])
        validate_mnemorag_full_optimizations(summaries[0])
        return summaries
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
    legacy_group, legacy_sources = legacy_group_sources(
        getattr(args, "reuse_campaign_root", None), group
    )
    if legacy_group is not None and not (group_dir / "workload.npz").exists():
        shutil.copy2(legacy_group / "workload.npz", group_dir / "workload.npz")
    attempts = sorted(group_dir.glob("attempt_*"))
    attempt_dir = group_dir / f"attempt_{len(attempts) + 1:03d}"
    attempt_dir.mkdir(parents=True)
    trace_path, selection_evidence = prepare_trace(
        group,
        group_dir,
        excluded_indices=history_indices,
        selection_root=getattr(args, "selection_root", None),
    )
    summaries = []
    summary_paths = []
    skipped = set(skip_systems or set())
    for record in group.records():
        system = record["system"]
        if system in skipped:
            continue
        if system in legacy_sources:
            summary, relative = materialize_legacy_baseline(
                group_dir, legacy_sources[system]
            )
            summaries.append(summary)
            summary_paths.append(relative)
            print(
                f"REUSE_LEGACY_BASELINE {group.identifier()} {system}",
                flush=True,
            )
            continue
        output_dir = attempt_dir / system
        command, cwd = build_system_command(
            args,
            group,
            record,
            trace_path,
            output_dir,
            mnemo_extra_args,
            selection_evidence,
        )
        if system == "mnemorag":
            reusable = reusable_mnemo_source(
                getattr(args, "reuse_mnemo_campaign_root", None),
                group,
                trace_path,
                command,
                selection_evidence,
            )
            if reusable is not None:
                summary, relative = materialize_reused_mnemo(group_dir, reusable)
                summaries.append(summary)
                summary_paths.append(relative)
                print(
                    f"REUSE_HASH_VERIFIED_MNEMO {group.identifier()}",
                    flush=True,
                )
                continue
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
        if system == "mnemorag" and args.next_use_profile_root is not None:
            profile_validation = validate_profile_guided_summary(
                summary,
                next_use_profile_path(args.next_use_profile_root, group),
            )
            atomic_write_json(
                summary_path.parent / "profile_guided_validation.json",
                profile_validation,
            )
            summary["profile_guided_validation"] = profile_validation
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
    full_split = group.system_order == ("mnemorag",)
    if full_split:
        source_count = int(mnemo_summary["source_queries"])
        excluded = set(map(int, history_indices))
        observed = list(map(int, mnemo_summary["sample_indices"]))
        if len(observed) != len(set(observed)):
            raise ValueError(f"{group.identifier()}: duplicate full-split indices")
        if set(observed) & excluded:
            raise ValueError(f"{group.identifier()}: full split overlaps history")
        if set(observed) | excluded != set(range(source_count)):
            raise ValueError(
                f"{group.identifier()}: full split does not cover every available row"
            )
        mnemo_summary["evaluation_scope"] = "full_split_long_horizon"
        mnemo_summary["split_disclosure"] = {
            "source_rows": source_count,
            "frozen_history_excluded_rows": len(excluded),
            "measured_rows": len(observed),
            "definition": (
                "complete available evaluation split after frozen-history exclusion"
            ),
        }
        mnemo_summary["provenance"] = {
            "classification": "current_semantics",
            "legacy_semantics": False,
            "post_arrival_secondary_aggregation_wait": False,
        }
        mnemo_summary["headline_eligible"] = True
        mnemo_summary["matched_cross_system_comparison_eligible"] = False
        atomic_write_json(group_dir / summary_paths[0], mnemo_summary)
        validation = validate_standalone_point(mnemo_summary)
    elif len(summaries) == len(group.system_order):
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
                validate_standalone_point(summary) for summary in summaries
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
                "full_split_long_horizon"
                if full_split
                else (
                    "matched_fair_subset"
                    if len(summaries) == len(group.system_order)
                    else "slo_censored_partial"
                )
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
            "workload_selection": selection_evidence,
        },
    )
    return summaries


def main() -> None:
    args = parse_args()
    if args.next_use_profile_root is not None:
        raise ValueError(
            "formal online serving must not receive a future next-use profile"
        )
    if args.selection_root is not None and args.reuse_campaign_root is not None:
        raise ValueError(
            "legacy candidate reuse is incompatible with an audited fixed selection"
        )
    if args.selection_root is not None and args.fixed_request_count is None:
        raise ValueError("--selection-root requires --fixed-request-count")
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
        if args.next_use_profile_root is not None:
            profile_path = next_use_profile_path(args.next_use_profile_root, group)
            if not profile_path.is_file():
                raise FileNotFoundError(
                    f"missing next-use profile for {group.identifier()}: {profile_path}"
                )
            group_environment.update(
                {
                    "PYTHONHASHSEED": "0",
                    "VLLM_KV_PROFILE_NEXT_USE_PATH": str(profile_path),
                    "VLLM_KV_PROFILE_OBSERVE_ONLY": "1",
                }
            )
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
