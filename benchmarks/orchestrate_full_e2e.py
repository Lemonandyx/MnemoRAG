#!/usr/bin/env python3
"""Orchestrate the complete, resumable MnemoRAG comparison campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

try:
    from .campaign_statistics import aggregate_repeated_groups, robust_slo_pass
    from .run_full_e2e_campaign import DEFAULT_PYTHON, atomic_write_json
    from .slo_campaign import DATASETS, QPS_GRIDS, SYSTEMS, WORKFLOWS
    from .source_provenance import capture_source_provenance
except ImportError:
    from campaign_statistics import aggregate_repeated_groups, robust_slo_pass
    from run_full_e2e_campaign import DEFAULT_PYTHON, atomic_write_json
    from slo_campaign import DATASETS, QPS_GRIDS, SYSTEMS, WORKFLOWS
    from source_provenance import capture_source_provenance


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HEDRA_ROOT = Path(
    os.environ.get("HEDRARAG_ROOT", PROJECT_ROOT / "third_party/HedraRAG")
).resolve()
VLLM_ROOT = Path(
    os.environ.get("MNEMORAG_VLLM_ROOT", PROJECT_ROOT / "third_party/vllm")
).resolve()
NUMACTL = ("numactl", "--physcpubind=0-63", "--interleave=0-3")
CPU_KV_CAPACITY_DECIMAL_GB = 96
CPU_KV_PINNED_TIER_DECIMAL_GB = 68
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_SELECTION_ROOT = PROJECT_ROOT / "results/evaluation/prefix_equivalent_workload"

SELECTION_CAMPAIGN_FORMAT = "mnemorag.prefix-equivalent-subset-campaign.v4"
ONLINE_CAUSAL_RUNTIME_GATE_FORMAT = "mnemorag.online-causal-proxy-runtime-gate.v5"


def normalize_formal_profile(profile: dict) -> dict:
    """Freeze capacity while keeping the pinned allocation below 64 GiB."""
    profile = dict(profile)
    arguments = [str(value) for value in profile.get("extra_args", [])]
    required_options = {
        "--cpu-kv-cache-gib": CPU_KV_CAPACITY_DECIMAL_GB * 10**9 / 1024**3,
        "--cpu-kv-pinned-tier-gib": (CPU_KV_PINNED_TIER_DECIMAL_GB * 10**9 / 1024**3),
    }
    for option, value in required_options.items():
        if option in arguments:
            arguments[arguments.index(option) + 1] = str(value)
        else:
            arguments.extend([option, str(value)])
    profile["extra_args"] = arguments
    resource_contract = dict(profile.get("resource_contract") or {})
    resource_contract.update(
        {
            "physical_cpu_ids": "0-63",
            "physical_core_count": 64,
            "socket_ids": [0],
            "numa_node_ids": [0, 1, 2, 3],
            "numa_policy": "interleave",
            "cpu_kv_capacity_decimal_gb": CPU_KV_CAPACITY_DECIMAL_GB,
            "cpu_kv_pinned_tier_decimal_gb": (CPU_KV_PINNED_TIER_DECIMAL_GB),
        }
    )
    profile["resource_contract"] = resource_contract
    return profile


def project_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    entries = [str(VLLM_ROOT), str(PROJECT_ROOT)]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mnemo-profile-template", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--history-size", type=int, default=2048)
    parser.add_argument("--history-seed", type=int, default=20260817)
    parser.add_argument(
        "--history-nprobe",
        type=int,
        default=128,
        help=(
            "IVF nprobe used to build the fixed warm-history snapshot. "
            "Existing history is reused only when this value, top-k, and "
            "history size all match."
        ),
    )
    parser.add_argument("--dataset-qps", type=float, default=4.0)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Shared vLLM context limit for every system and campaign phase.",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=DEFAULT_SELECTION_ROOT,
        help="Audited fixed-size workload shared by every compared system.",
    )
    parser.add_argument("--request-count", type=int, default=512)
    parser.add_argument(
        "--measurement-repeats",
        type=int,
        default=1,
        help=(
            "Number of measurements per selected final-SLO, fixed-nprobe, "
            "and ablation point."
        ),
    )
    parser.add_argument(
        "--reuse-candidate-as-final-slo",
        action="store_true",
        help=(
            "Use the single candidate QPS scan as the final SLO evidence "
            "instead of rerunning selected boundary points."
        ),
    )
    parser.add_argument(
        "--reuse-candidate-root",
        type=Path,
        help="Accepted candidate phase root to import as legacy_semantics evidence.",
    )
    parser.add_argument(
        "--reuse-mnemo-candidate-root",
        type=Path,
        help=(
            "Reuse only command-, workload-, and source-hash-identical "
            "MnemoRAG candidate points. Baselines are remeasured."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument(
        "--resource-policy",
        choices=("strict", "expedite"),
        default="strict",
        help=(
            "strict enforces the formal 5%% CPU gate; expedite records resource "
            "evidence while only blocking on target-GPU conflicts"
        ),
    )
    parser.add_argument(
        "--stop-after-slo-violation",
        action="store_true",
        help=(
            "Retain each system's first candidate point with p99 above the "
            "10-second SLO, then stop that system at higher QPS values."
        ),
    )
    return parser.parse_args()


def command_signature(log_path: Path) -> str:
    if not log_path.exists():
        return "missing-log"
    tail = log_path.read_bytes()[-32_768:].decode("utf-8", errors="replace")
    # Retry output paths and process IDs change on every attempt even when the
    # underlying failure is identical. Normalize only those volatile fields so
    # deterministic point failures stop after the existing three-strike limit.
    tail = re.sub(r"attempt_\d+", "attempt_N", tail)
    tail = re.sub(r"\bPID\s+\d+\b", "PID N", tail)
    return hashlib.sha256(tail.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_resource_gate(args: argparse.Namespace, label: str) -> None:
    gate_dir = args.output_dir / "resource_gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_path = gate_dir / f"{label}.json"
    expedited = args.resource_policy == "expedite"
    command = [
        *NUMACTL,
        str(args.python),
        str(HERE / "wait_for_formal_resources.py"),
        "--gpu-id",
        "1",
        "--max-cpu-busy-percent",
        "100" if expedited else "5",
        "--max-gpu-memory-mib",
        "256",
        "--max-gpu-util-percent",
        "5",
        "--consecutive-clean-samples",
        "1" if expedited else "3",
        "--retry-seconds",
        str(min(args.retry_seconds, 5.0) if expedited else args.retry_seconds),
        "--output",
        str(gate_path),
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=project_environment(),
        check=True,
    )


def run_resumable(
    args: argparse.Namespace,
    label: str,
    command: list[str],
) -> None:
    logs = args.output_dir / "driver_logs"
    logs.mkdir(parents=True, exist_ok=True)
    repeated_signature = None
    repeated_count = 0
    for attempt in range(1, args.max_attempts + 1):
        run_resource_gate(args, f"{label}_attempt_{attempt:02d}")
        log_path = logs / f"{label}_attempt_{attempt:02d}.log"
        atomic_write_json(
            logs / f"{label}_attempt_{attempt:02d}.command.json",
            {"argv": command, "cwd": str(PROJECT_ROOT)},
        )
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=project_environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode == 0:
            return
        signature = command_signature(log_path)
        repeated_count = repeated_count + 1 if signature == repeated_signature else 1
        repeated_signature = signature
        if repeated_count >= 3:
            raise RuntimeError(
                f"{label} failed three times with the same log tail; see {log_path}"
            )
        time.sleep(args.retry_seconds)
    raise RuntimeError(f"{label} exceeded {args.max_attempts} attempts")


def build_histories(args: argparse.Namespace) -> Path:
    if args.history_size <= 0:
        raise ValueError("--history-size must be positive")
    if args.history_nprobe <= 0:
        raise ValueError("--history-nprobe must be positive")
    profile_path = args.output_dir / "mnemo_formal_profile.json"
    history_root = args.output_dir / "history"
    if not profile_path.exists():
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.mnemo_profile_template, profile_path)
    profile = normalize_formal_profile(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    removed_datasets = {}
    for field in ("dataset_history", "dataset_environment"):
        values = profile.get(field) or {}
        removed_datasets[field] = sorted(set(values) - set(DATASETS))
        profile[field] = {
            dataset: values[dataset] for dataset in DATASETS if dataset in values
        }
    atomic_write_json(profile_path, profile)
    atomic_write_json(
        args.output_dir / "formal_dataset_selection.json",
        {
            "datasets": list(DATASETS),
            "removed_from_new_profile": removed_datasets,
            "historical_artifacts_preserved": True,
        },
    )
    for dataset in DATASETS:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        history = (profile.get("dataset_history") or {}).get(dataset)
        if history is not None:
            required_history_contract = {
                "history_count": int(args.history_size),
                "retrieval_nprobe": int(args.history_nprobe),
                "retrieval_topk": 3,
            }
            artifact_pairs = (
                ("sample_indices_path", "sample_indices_sha256"),
                ("prior_path", "prior_sha256"),
                ("query_base_snapshot_path", "query_base_snapshot_sha256"),
                ("question_hashes_path", "question_hashes_sha256"),
            )
            reusable = all(
                int(history.get(key, -1)) == expected
                for key, expected in required_history_contract.items()
            )
            evidence = {}
            for path_key, hash_key in artifact_pairs:
                if not reusable:
                    break
                path = Path(str(history.get(path_key, "")))
                expected = str(history.get(hash_key, ""))
                if not path.is_file() or not expected:
                    reusable = False
                    break
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
                if observed != expected:
                    reusable = False
                    break
                evidence[path_key] = {
                    "path": str(path.resolve()),
                    "sha256": observed,
                }
            if reusable:
                atomic_write_json(
                    history_root / dataset / "history_reuse_manifest.json",
                    {
                        "status": "reused",
                        "dataset": dataset,
                        "provenance": "existing_hash_verified_history",
                        "required_history_contract": required_history_contract,
                        "artifacts": evidence,
                    },
                )
                continue
        command = [
            *NUMACTL,
            str(args.python),
            str(HERE / "prepare_disjoint_history.py"),
            "--dataset",
            dataset,
            "--history-size",
            str(args.history_size),
            "--seed",
            str(args.history_seed),
            "--nprobe",
            str(args.history_nprobe),
            "--topk",
            "3",
            "--retrieval-batch-size",
            "64",
            "--minimum-frequency",
            "2",
            "--gpu-id",
            "1",
            "--cpu-threads",
            "64",
            "--faiss-omp-threads",
            "1",
            "--faiss-search-workers",
            "64",
            "--require-clean-resources",
            "--output-dir",
            str(history_root),
            "--profile-template",
            str(profile_path),
            "--profile-output",
            str(profile_path),
        ]
        if args.resource_policy == "expedite":
            command.extend(
                [
                    "--max-preflight-cpu-busy-percent",
                    "100",
                    "--max-preflight-cpu-max-busy-percent",
                    "100",
                    "--max-runtime-external-cpu-busy-percent",
                    "100",
                ]
            )
        run_resumable(args, f"history_{dataset}", command)
    return profile_path


def validate_online_causal_selection(selection_root: Path) -> dict:
    selection_root = Path(selection_root).resolve()
    selection_campaign = selection_root / "selection_campaign_manifest.json"
    if not selection_campaign.is_file():
        raise ValueError(f"missing audited selection campaign: {selection_campaign}")
    selection_metadata = json.loads(selection_campaign.read_text(encoding="utf-8"))
    if selection_metadata.get("format") != SELECTION_CAMPAIGN_FORMAT:
        raise ValueError(
            "unsupported selection campaign format: "
            f"{selection_metadata.get('format')!r}"
        )
    if selection_metadata.get("all_validations_passed") is not True:
        raise ValueError("selection campaign did not pass all offline validations")
    if selection_metadata.get("runtime_mechanism_gate_required") is not True:
        raise ValueError(
            "selection campaign does not require the runtime mechanism gate"
        )
    if set(selection_metadata.get("datasets") or []) != set(DATASETS):
        raise ValueError("selection campaign does not contain the four formal datasets")
    if int(selection_metadata.get("selected_rows_per_dataset", -1)) != 512:
        raise ValueError("selection campaign is not fixed to 512 rows per dataset")

    mechanism_gate = selection_root / "profile_observed_four_dataset_runtime_gate.json"
    if not mechanism_gate.is_file():
        raise ValueError(f"missing runtime mechanism gate: {mechanism_gate}")
    mechanism_metadata = json.loads(mechanism_gate.read_text(encoding="utf-8"))
    if (
        mechanism_metadata.get("format") != ONLINE_CAUSAL_RUNTIME_GATE_FORMAT
        or mechanism_metadata.get("status") != "passed"
        or mechanism_metadata.get("complete_four_dataset_gate") is not True
        or mechanism_metadata.get("runtime_future_profile_used") is not False
        or {str(row.get("dataset")) for row in mechanism_metadata.get("datasets") or []}
        != set(DATASETS)
    ):
        raise ValueError("runtime mechanism gate did not pass")
    if (
        Path(str(mechanism_metadata.get("selection_campaign_path", ""))).resolve()
        != selection_campaign.resolve()
        or mechanism_metadata.get("selection_campaign_sha256")
        != file_sha256(selection_campaign)
        or "no future next-use data is used"
        not in str(mechanism_metadata.get("semantics", ""))
    ):
        raise ValueError("runtime mechanism gate uses another selection")
    resource_contract = mechanism_metadata.get("resource_contract") or {}
    if (
        resource_contract.get("physical_cpu_ids") != "0-63"
        or int(resource_contract.get("physical_core_count", -1)) != 64
        or resource_contract.get("socket_ids") != [0]
        or resource_contract.get("numa_node_ids") != [0, 1, 2, 3]
        or int(resource_contract.get("physical_gpu_id", -1)) != 1
    ):
        raise ValueError("runtime mechanism gate has the wrong hardware topology")
    mechanism_rows = {
        str(row.get("dataset")): row for row in mechanism_metadata.get("datasets") or []
    }
    for dataset in DATASETS:
        row = mechanism_rows[dataset]
        selection_manifest = selection_root / dataset / "selection_manifest.json"
        validation = row.get("validation") or {}
        checks = validation.get("checks") or {}
        summary_path = Path(str(row.get("summary_path", ""))).resolve()
        if (
            row.get("status") != "passed"
            or validation.get("status") != "passed"
            or not checks
            or any(value is not True for value in checks.values())
            or validation.get("profile_guided") is not False
            or int(validation.get("profile_accesses", -1)) != 0
            or int(validation.get("profile_matches", -1)) != 0
            or int(validation.get("profile_finite_accesses", -1)) != 0
            or Path(str(row.get("selection_manifest_path", ""))).resolve()
            != selection_manifest.resolve()
            or row.get("selection_manifest_sha256") != file_sha256(selection_manifest)
            or not summary_path.is_file()
            or row.get("summary_sha256") != file_sha256(summary_path)
        ):
            raise ValueError(f"runtime mechanism evidence mismatch for {dataset}")
    return {
        "selection_campaign": str(selection_campaign),
        "selection_campaign_sha256": file_sha256(selection_campaign),
        "runtime_mechanism_gate": str(mechanism_gate),
        "runtime_mechanism_gate_sha256": file_sha256(mechanism_gate),
        "runtime_future_profile_used": False,
    }


def run_campaign_phases(args: argparse.Namespace, profile_path: Path) -> None:
    candidate = args.output_dir / "candidate"
    final_slo = args.output_dir / "final_slo"
    dataset_latency = args.output_dir / "dataset_latency"
    if args.request_count != 512:
        raise ValueError("the formal proxy campaign is fixed to 512 requests")
    if args.measurement_repeats <= 0:
        raise ValueError("--measurement-repeats must be positive")
    if args.reuse_candidate_root is not None:
        raise ValueError(
            "legacy candidate results cannot be reused with the new fixed selection"
        )
    gate_evidence = validate_online_causal_selection(args.selection_root)
    selection_campaign = Path(gate_evidence["selection_campaign"])
    mechanism_gate = Path(gate_evidence["runtime_mechanism_gate"])
    atomic_write_json(
        args.output_dir / "preflight_gate.json",
        {
            "status": "passed",
            "semantics": "online_causal_poisson_serving",
            "future_next_use_profile_used": False,
            "runtime_mechanism_gate": str(mechanism_gate.resolve()),
            "runtime_mechanism_gate_sha256": file_sha256(mechanism_gate),
        },
    )
    atomic_write_json(
        args.output_dir / "workload_selection.json",
        {
            "evaluation_scope": "sar_and_prefix_reuse_equivalent_512_proxy",
            "full_split_runs_enabled": False,
            "request_count_per_point": args.request_count,
            "measurement_repeats": args.measurement_repeats,
            "candidate_scan_is_final_slo": bool(args.reuse_candidate_as_final_slo),
            "candidate_slo_early_stop": {
                "enabled": bool(args.stop_after_slo_violation),
                "metric": "request_latency_p99_s",
                "threshold_seconds": 10.0,
                "maximum_qps_by_workflow": {
                    workflow: max(QPS_GRIDS[workflow]) for workflow in WORKFLOWS
                },
                "extrapolation_beyond_qps_cap": False,
                "semantics": (
                    "retain the first violating point per system and skip only "
                    "that system at higher QPS; unmatched missing points are "
                    "right-censored, never imputed; if no violation is observed "
                    "through QPS 64, report capacity as a lower bound of at "
                    "least 64 QPS and do not extend the scan"
                ),
            },
            "reused_mnemo_candidate_root": (
                str(args.reuse_mnemo_candidate_root.resolve())
                if args.reuse_mnemo_candidate_root is not None
                else None
            ),
            "history_retrieval_nprobe": args.history_nprobe,
            "history_retrieval_topk": 3,
            "history_size": args.history_size,
            "selection_root": str(args.selection_root.resolve()),
            "selection_campaign_manifest": str(selection_campaign.resolve()),
            "selection_campaign_manifest_sha256": hashlib.sha256(
                selection_campaign.read_bytes()
            ).hexdigest(),
            "runtime_mechanism_gate": str(mechanism_gate.resolve()),
            "runtime_mechanism_gate_sha256": hashlib.sha256(
                mechanism_gate.read_bytes()
            ).hexdigest(),
            "runtime_cache_evidence_role": {
                "semantics": "online_causal_poisson_serving",
                "future_next_use_profile_used": False,
                "causal_document_hotness": True,
                "purpose": (
                    "verify that the full-trace-derived 512-query proxy "
                    "reproduces prefix reuse using only already-observed requests"
                ),
            },
            "cross_system_requirement": (
                "every system replays the same selected indices and arrival trace"
            ),
        },
    )
    common = [
        "--mnemo-profile",
        str(profile_path),
        "--python",
        str(args.python),
        "--gpu-id",
        "1",
        "--gpu-memory-utilization",
        "0.55",
        "--retrieval-batch-size",
        "64",
        "--serving-batch-size",
        "64",
        "--max-num-batched-tokens",
        "12288",
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        "32",
        "--latency-slo-seconds",
        "10",
        "--resource-policy",
        args.resource_policy,
        "--fixed-request-count",
        str(args.request_count),
        "--selection-root",
        str(args.selection_root),
    ]
    candidate_command = [
        *NUMACTL,
        str(args.python),
        str(HERE / "run_full_e2e_campaign.py"),
        "--phase",
        "candidate",
        "--output-dir",
        str(candidate),
        *common,
    ]
    if args.reuse_mnemo_candidate_root is not None:
        candidate_command.extend(
            [
                "--reuse-mnemo-campaign-root",
                str(args.reuse_mnemo_candidate_root),
            ]
        )
    if args.stop_after_slo_violation:
        candidate_command.append("--stop-after-slo-violation")
    run_resumable(args, "candidate", candidate_command)
    candidate_results = candidate / "candidate_results.json"

    if args.reuse_candidate_as_final_slo:
        final_slo.mkdir(parents=True, exist_ok=True)
        final_results = final_slo / "final-slo_results.json"
        shutil.copyfile(candidate_results, final_results)
        censoring_path = candidate / "candidate_slo_censoring.json"
        if censoring_path.is_file():
            shutil.copyfile(
                censoring_path,
                final_slo / "candidate_slo_censoring.json",
            )
        atomic_write_json(
            final_slo / "candidate_scan_reuse.json",
            {
                "status": "reused_without_rerun",
                "source": str(candidate_results.resolve()),
                "destination": str(final_results.resolve()),
                "reason": "explicit single-measurement expedited campaign",
                "statistical_disclosure": (
                    "Each QPS point is a single measurement; no boundary "
                    "confirmation repeat was performed."
                ),
            },
        )
    else:
        final_command = [
            *NUMACTL,
            str(args.python),
            str(HERE / "run_full_e2e_campaign.py"),
            "--phase",
            "final-slo",
            "--output-dir",
            str(final_slo),
            "--candidate-results",
            str(candidate_results),
            "--repeats",
            str(args.measurement_repeats),
            *common,
        ]
        run_resumable(args, "final_slo", final_command)

    dataset_command = [
        *NUMACTL,
        str(args.python),
        str(HERE / "run_full_e2e_campaign.py"),
        "--phase",
        "dataset-latency",
        "--output-dir",
        str(dataset_latency),
        "--dataset-qps",
        str(args.dataset_qps),
        "--dataset-nprobes",
        "256",
        "512",
        "--repeats",
        str(args.measurement_repeats),
        *common,
    ]
    run_resumable(args, "dataset_latency", dataset_command)


def select_ablation_qps(final_results_path: Path, *, minimum_repeats: int = 2) -> float:
    summaries = json.loads(final_results_path.read_text(encoding="utf-8"))
    aggregated = aggregate_repeated_groups(
        summaries,
        minimum_repeats=minimum_repeats,
        allow_incomplete_system_groups=True,
    )
    candidates = [
        row
        for row in aggregated
        if row["system"] == "mnemorag"
        and row.get("dataset", "nq") == "nq"
        and row["workflow"] == "onepass"
        and row["nprobe"] == 128
    ]
    passing = [row for row in candidates if robust_slo_pass(row)]
    selected = max(passing or candidates, key=lambda row: row["arrival_rate_rps"])
    return float(selected["arrival_rate_rps"])


def run_analysis(args: argparse.Namespace, profile_path: Path) -> None:
    final_results = args.output_dir / "final_slo" / "final-slo_results.json"
    dataset_results = (
        args.output_dir / "dataset_latency" / "dataset-latency_results.json"
    )
    analysis = args.output_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name, source in (
        ("final_slo", final_results),
        ("dataset_latency", dataset_results),
    ):
        command = [
            *NUMACTL,
            str(args.python),
            str(HERE / "quality_noninferiority.py"),
            str(source),
            str(analysis / f"{name}_quality_noninferiority.json"),
            "--bootstrap-samples",
            "10000",
        ]
        run_resumable(args, f"quality_{name}", command)

    ablation_qps = select_ablation_qps(
        final_results, minimum_repeats=args.measurement_repeats
    )
    atomic_write_json(
        analysis / "ablation_selection.json",
        {
            "dataset": "nq",
            "workflow": "onepass",
            "nprobe": 128,
            "arrival_rate_rps": ablation_qps,
            "selection": "highest robust MnemoRAG 10s-SLO point",
        },
    )
    ablation_command = [
        *NUMACTL,
        str(args.python),
        str(HERE / "run_mnemo_ablation.py"),
        "--output-dir",
        str(args.output_dir / "ablation"),
        "--mnemo-profile",
        str(profile_path),
        "--python",
        str(args.python),
        "--dataset",
        "nq",
        "--workflow",
        "onepass",
        "--qps",
        str(ablation_qps),
        "--nprobe",
        "128",
        "--max-model-len",
        str(args.max_model_len),
        "--num-requests",
        str(args.request_count),
        "--selection-root",
        str(args.selection_root),
        "--repeats",
        str(args.measurement_repeats),
        "--resource-policy",
        args.resource_policy,
    ]
    run_resumable(args, "ablation", ablation_command)
    summarize_command = [
        *NUMACTL,
        str(args.python),
        str(HERE / "summarize_full_e2e.py"),
        "--campaign-root",
        str(args.output_dir),
        "--output-dir",
        str(analysis / "figures_and_tables"),
        "--minimum-repeats",
        str(args.measurement_repeats),
    ]
    run_resumable(args, "summarize", summarize_command)


def validate_final_deliverables(args: argparse.Namespace) -> dict:
    stop_after_slo_violation = bool(getattr(args, "stop_after_slo_violation", False))
    raw_paths = {
        "candidate": args.output_dir / "candidate" / "candidate_results.json",
        "final_slo": args.output_dir / "final_slo" / "final-slo_results.json",
        "dataset_latency": (
            args.output_dir / "dataset_latency" / "dataset-latency_results.json"
        ),
        "ablation": args.output_dir / "ablation" / "ablation_results.json",
    }
    candidate_run_count = (
        len(DATASETS)
        * sum(len(QPS_GRIDS[workflow]) for workflow in WORKFLOWS)
        * len(SYSTEMS)
    )
    expected_counts = {
        "candidate": (None if stop_after_slo_violation else candidate_run_count),
        "final_slo": (
            (None if stop_after_slo_violation else candidate_run_count)
            if args.reuse_candidate_as_final_slo
            else None
        ),
        "dataset_latency": 72 * args.measurement_repeats,
        "ablation": 7 * args.measurement_repeats,
    }
    raw_evidence = {}
    for label, path in raw_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing raw result set: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        expected = expected_counts[label]
        if expected is not None and len(rows) != expected:
            raise RuntimeError(f"{label} has {len(rows)} rows; expected {expected}")
        if not rows:
            raise RuntimeError(f"empty raw result set: {label}")
        if (
            stop_after_slo_violation
            and label in {"candidate", "final_slo"}
            and len(rows) > candidate_run_count
        ):
            raise RuntimeError(
                f"{label} has {len(rows)} rows; maximum is {candidate_run_count}"
            )
        expected_datasets = {"nq"} if label == "ablation" else set(DATASETS)
        if {row.get("dataset") for row in rows} != expected_datasets:
            raise RuntimeError(
                f"{label} dataset coverage differs from {sorted(expected_datasets)}"
            )
        expected_systems = (
            {"mnemorag"}
            if label == "ablation"
            else {"mnemorag", "flashrag", "hedrarag"}
        )
        if {row.get("system") for row in rows} != expected_systems:
            raise RuntimeError(f"{label} does not cover all three systems")
        if any(int(row.get("num_requests", -1)) != args.request_count for row in rows):
            raise RuntimeError(f"{label} contains a non-{args.request_count} point")
        if label != "ablation" and any(
            row.get("headline_eligible") is not False
            for row in rows
            if row.get("system") == "hedrarag"
        ):
            raise RuntimeError(f"{label} contains headline-eligible HedraRAG data")
        raw_evidence[label] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "row_count": len(rows),
        }

    output = args.output_dir / "analysis" / "figures_and_tables"
    report_path = output / "final_report.json"
    if not report_path.is_file():
        raise RuntimeError("final report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("quality_noninferiority_all_pass") is not True
        or report.get("hedrarag_headline_eligible") is not False
        or int(report.get("measurement_repeat_count", -1)) != args.measurement_repeats
    ):
        raise RuntimeError("final report acceptance checks failed")

    required_tables = (
        "final_report.json",
        "final_slo_aggregated.json",
        "final_slo_aggregated.csv",
        "dataset_latency_aggregated.json",
        "dataset_latency_aggregated.csv",
        "ablation_aggregated.json",
        "slo_capacities.csv",
        "slo_capacity_speedups.csv",
        "dataset_latency_speedups.csv",
    )
    figure_stems = [
        "e2e_mean_latency_qps_3x4",
        *(f"robust_slo_capacity_{dataset}" for dataset in DATASETS),
        "multidataset_e2e_latency",
        "mnemorag_component_ablation",
    ]
    required_artifacts = [output / name for name in required_tables]
    required_artifacts.extend(
        output / f"{stem}.{suffix}"
        for stem in figure_stems
        for suffix in ("png", "pdf", "svg")
    )
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing final artifacts: {missing}")
    if not list((args.output_dir / "driver_logs").glob("*.command.json")):
        raise RuntimeError("command provenance is missing")
    if not list((args.output_dir / "resource_gates").glob("*.json")):
        raise RuntimeError("resource evidence is missing")
    return {
        "status": "passed",
        "raw_results": raw_evidence,
        "final_report": {
            "path": str(report_path.resolve()),
            "sha256": file_sha256(report_path),
        },
        "artifact_count": len(required_artifacts),
        "artifacts": {
            str(path.relative_to(args.output_dir)): file_sha256(path)
            for path in required_artifacts
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_start = capture_source_provenance(
        PROJECT_ROOT,
        HEDRA_ROOT,
        VLLM_ROOT,
    )
    atomic_write_json(args.output_dir / "source_provenance_start.json", source_start)
    profile_path = build_histories(args)
    run_campaign_phases(args, profile_path)
    run_analysis(args, profile_path)
    source_end = capture_source_provenance(
        PROJECT_ROOT,
        HEDRA_ROOT,
        VLLM_ROOT,
    )
    atomic_write_json(args.output_dir / "source_provenance_end.json", source_end)
    if source_end["combined_source_sha256"] != source_start["combined_source_sha256"]:
        raise RuntimeError(
            "runtime source files changed during the formal campaign; results "
            "cannot be marked complete"
        )
    deliverables = validate_final_deliverables(args)
    atomic_write_json(
        args.output_dir / "campaign_complete.json",
        {
            "status": "complete",
            "completed_at_unix_s": time.time(),
            "profile": str(profile_path.resolve()),
            "combined_source_sha256": source_start["combined_source_sha256"],
            "source_provenance_start": str(
                (args.output_dir / "source_provenance_start.json").resolve()
            ),
            "source_provenance_end": str(
                (args.output_dir / "source_provenance_end.json").resolve()
            ),
            "deliverable_validation": deliverables,
        },
    )
    print("FULL_E2E_CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
