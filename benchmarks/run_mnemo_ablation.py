#!/usr/bin/env python3
"""Run the repeated, request-matched MnemoRAG component ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

try:
    from .comparison_validation import validate_mnemorag_full_optimizations
    from .run_full_e2e_campaign import (
        DEFAULT_PYTHON,
        atomic_write_json,
        build_system_command,
        find_summary,
        load_history_indices,
        load_profile,
        prepare_trace,
        verify_driver_topology,
    )
    from .slo_campaign import PointGroup
except ImportError:
    from comparison_validation import validate_mnemorag_full_optimizations
    from run_full_e2e_campaign import (
        DEFAULT_PYTHON,
        atomic_write_json,
        build_system_command,
        find_summary,
        load_history_indices,
        load_profile,
        prepare_trace,
        verify_driver_topology,
    )
    from slo_campaign import PointGroup


VARIANTS = (
    "full",
    "no_sar",
    "no_hotness",
    "no_async_pipeline",
    "no_cpu_kv",
    "no_cache_aware_prefill",
    "no_history_warmstart",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mnemo-profile", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--workflow", default="onepass")
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument(
        "--selection-root",
        type=Path,
        help="Audited fixed workload used by the main comparison campaign.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--serving-batch-size", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=12288)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--latency-slo-seconds", type=float, default=10.0)
    parser.add_argument(
        "--resource-policy",
        choices=("strict", "expedite"),
        default="strict",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def _set_option(command: list[str], option: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == option]
    if not positions:
        command.extend([option, value])
        return
    for position in positions:
        command[position + 1] = value


def apply_variant(command: list[str], variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported ablation variant: {variant}")
    command = list(command)
    command.extend(["--campaign-variant", variant])
    if variant == "no_sar":
        _set_option(command, "--components", "kv_full")
    elif variant == "no_hotness":
        _set_option(command, "--components", "full_no_hotness")
        _set_option(command, "--order-mode", "original")
        command.extend(
            [
                "--no-document-kv-heat",
                "--kv-policy-override",
                "lru",
                "--cpu-kv-eviction-policy",
                "lru",
                "--cpu-kv-min-document-heat-to-store",
                "0",
            ]
        )
    elif variant == "no_async_pipeline":
        _set_option(command, "--components", "full_no_async")
        _set_option(command, "--async-retrieval-policy", "sequential")
        command.extend(
            ["--no-cross-microbatch-pipeline", "--no-cache-aware-prefill-batching"]
        )
    elif variant == "no_cpu_kv":
        command = [
            item
            for item in command
            if item
            not in {"--require-cpu-kv-h2d-samples", "--require-cpu-kv-d2h-samples"}
        ]
        _set_option(command, "--cpu-kv-cache-gib", "0")
    elif variant == "no_cache_aware_prefill":
        command.append("--no-cache-aware-prefill-batching")
    return command


def variant_environment(
    base: dict[str, str],
    variant: str,
) -> dict[str, str]:
    environment = dict(base)
    environment.pop("VLLM_KV_PROFILE_NEXT_USE_PATH", None)
    environment.pop("VLLM_KV_ORACLE_NEXT_USE_PATH", None)
    environment.pop("VLLM_KV_PROFILE_RECORD_PATH", None)
    environment.pop("VLLM_KV_PROFILE_OBSERVE_ONLY", None)
    if variant == "no_history_warmstart":
        environment.pop("MNEMORAG_DOC_HOTNESS_PRIOR_PATH", None)
        environment.pop("MNEMORAG_QUERY_BASE_SNAPSHOT_PATH", None)
        environment["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] = "0"
    return environment


def validate_variant(
    summary: dict,
    variant: str,
) -> None:
    if summary.get("campaign_variant") != variant:
        raise RuntimeError(f"variant label mismatch for {variant}")
    if summary.get("status") != "complete":
        raise RuntimeError(f"incomplete ablation result: {variant}")
    topology = summary.get("cpu_topology") or {}
    if (
        topology.get("physical_core_count") != 64
        or topology.get("socket_ids") != [0]
        or topology.get("numa_node_ids") != [0, 1, 2, 3]
        or topology.get("smt_siblings_selected") is not False
    ):
        raise RuntimeError(f"invalid CPU topology for {variant}")
    monitor = summary.get("runtime_resource_monitor") or {}
    if monitor.get("status") != "clean" or monitor.get("violations") != []:
        raise RuntimeError(f"resource contamination in {variant}")
    if variant == "full":
        validate_mnemorag_full_optimizations(summary)
    profile_metadata = summary.get("profile_guided_online_serving") or {}
    profile_stats = summary.get("kv_cache_stats_total_including_warmup") or {}
    if (
        profile_metadata.get("enabled") is True
        or int(profile_stats.get("profile_next_use_accesses", 0) or 0) != 0
        or int(profile_stats.get("profile_next_use_matches", 0) or 0) != 0
    ):
        raise RuntimeError(f"future next-use profile was active for {variant}")
    if variant == "no_sar" and summary.get("enable_similarity_aware") is not False:
        raise RuntimeError("no_sar did not disable similarity-aware retrieval")
    elif variant == "no_hotness" and (
        summary.get("enable_hotness_sort") is not False
        or summary.get("document_kv_heat") is not False
        or summary.get("kv_policy") != "lru"
    ):
        raise RuntimeError("no_hotness did not isolate the heat mechanism")
    elif (
        variant == "no_async_pipeline"
        and summary.get("enable_async_scheduling") is not False
    ):
        raise RuntimeError("no_async_pipeline did not disable async scheduling")
    elif (
        variant == "no_cpu_kv"
        and (summary.get("cpu_kv_offload") or {}).get("enabled") is not False
    ):
        raise RuntimeError("no_cpu_kv did not disable host KV offload")
    elif (
        variant == "no_cache_aware_prefill"
        and summary.get("cache_aware_prefill_batching") is not False
    ):
        raise RuntimeError("no_cache_aware_prefill did not disable batching")
    elif variant == "no_history_warmstart" and (
        (summary.get("document_hotness_prior") or {}).get("enabled") is not False
        or (summary.get("query_base_snapshot") or {}).get("enabled") is not False
    ):
        raise RuntimeError("no_history_warmstart still loaded historical state")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    verify_driver_topology(args.gpu_id)
    extra_args, base_environment, profile = load_profile(
        args.mnemo_profile, args.dataset
    )
    excluded_indices, history = load_history_indices(profile, args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for repeat in range(args.repeats):
        group = PointGroup(
            phase="ablation",
            dataset=args.dataset,
            workflow=args.workflow,
            nprobe=args.nprobe,
            qps=args.qps,
            repeat=repeat,
            seed=args.seed + repeat,
            num_requests=args.num_requests,
            system_order=("mnemorag",),
        )
        repeat_dir = args.output_dir / f"repeat_{repeat}"
        trace_path, selection_evidence = prepare_trace(
            group,
            repeat_dir,
            excluded_indices=excluded_indices,
            selection_root=args.selection_root,
        )
        order = VARIANTS if repeat % 2 == 0 else tuple(reversed(VARIANTS))
        for position, variant in enumerate(order):
            variant_dir = repeat_dir / variant
            accepted = variant_dir / "accepted.json"
            if accepted.exists():
                summary_path = Path(
                    json.loads(accepted.read_text(encoding="utf-8"))["summary_path"]
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                validate_variant(summary, variant)
                all_summaries.append(summary)
                continue
            record = {"system": "mnemorag", "order_position": position}
            command, cwd = build_system_command(
                args,
                group,
                record,
                trace_path,
                variant_dir / "run",
                extra_args,
                selection_evidence,
            )
            command = apply_variant(command, variant)
            environment_overrides = variant_environment(
                base_environment,
                variant,
            )
            environment = os.environ.copy()
            environment.update(environment_overrides)
            variant_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                variant_dir / "command.json",
                {
                    "argv": command,
                    "cwd": str(cwd),
                    "environment_overrides": environment_overrides,
                    "workload_selection": selection_evidence,
                },
            )
            with (variant_dir / "driver.log").open("w", encoding="utf-8") as stream:
                subprocess.run(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True,
                )
            summary_path = find_summary(variant_dir / "run")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_variant(summary, variant)
            atomic_write_json(
                accepted,
                {
                    "status": "accepted",
                    "accepted_at_unix_s": time.time(),
                    "variant": variant,
                    "repeat": repeat,
                    "group": asdict(group),
                    "history": history,
                    "summary_path": str(summary_path.resolve()),
                    "online_cache_contract": {
                        "status": "online_causal",
                        "future_next_use_profile_used": False,
                        "document_hotness_evolves_from_observed_requests": True,
                    },
                    "resource_policy": {
                        "mode": args.resource_policy,
                        "cpu_busy_threshold_percent": (
                            100 if args.resource_policy == "expedite" else 5
                        ),
                        "gpu_exclusivity_required": True,
                        "resource_monitoring_enabled": True,
                    },
                },
            )
            all_summaries.append(summary)
            atomic_write_json(args.output_dir / "ablation_results.json", all_summaries)
    print(f"MNEMORAG_ABLATION_COMPLETE runs={len(all_summaries)}", flush=True)


if __name__ == "__main__":
    main()
