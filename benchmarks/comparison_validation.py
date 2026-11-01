"""Strict validation for fair MnemoRAG, FlashRAG, and HedraRAG groups."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .experiment_validation import expected_prompt_contract
except ImportError:
    from experiment_validation import expected_prompt_contract


REQUIRED_SYSTEMS = ("mnemorag", "flashrag", "hedrarag")
EXPECTED_CPU_IDS = list(range(64))
EXPECTED_SOCKET_IDS = [0]
EXPECTED_NUMA_NODE_IDS = [0, 1, 2, 3]
EXPECTED_NUMA_NODE_CORE_COUNTS = {str(node): 16 for node in range(4)}
EXPECTED_SOCKET_CORE_COUNTS = {"0": 64}
EXPECTED_ORDERING_MODES = {
    "mnemorag": "hotness_hint",
    "flashrag": "original",
    "hedrarag": "retrieval",
}
ORDERING_ONLY_PROMPT_FIELDS = {
    "ordering_mode",
    "document_order",
    "relevance_hint",
}
COMMON_POINT_FIELDS = (
    "dataset",
    "workflow",
    "num_requests",
    "seed",
    "arrival_rate_rps",
    "arrival_burstiness",
    "normalize_arrival_trace",
    "arrival_trace_sha256",
    "latency_slo_seconds",
    "campaign_group_id",
    "campaign_repeat",
    "topk",
    "nprobe",
    "retrieval_batch_size",
    "serving_batch_size",
    "max_num_batched_tokens",
    "vllm_engine_effective",
    "irg_rounds",
    "max_model_len",
    "max_tokens",
    "recomp_compression_max_tokens",
    "temperature",
    "ignore_eos",
    "gpu_memory_utilization",
    "prefix_caching",
    "model_path",
    "retrieval_model_path",
    "index_path",
    "corpus_path",
    "physical_gpu_id",
    "visible_gpu_count",
    "cpu_threads",
)


class ComparisonContractError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _expand_cpu_or_node_set(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return sorted(int(item) for item in value)
    result: set[int] = set()
    for token in re.split(r"[\s,]+", str(value or "").strip()):
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        elif token.isdigit():
            result.add(int(token))
    return sorted(result)


def _require_clean_resources(summary: Mapping[str, Any]) -> None:
    system = summary.get("system", "unknown")
    topology = summary.get("cpu_topology") or {}
    if topology.get("logical_cpu_ids") != EXPECTED_CPU_IDS:
        raise ComparisonContractError(
            f"{system}: CPU affinity must be exactly physical CPUs 0-63"
        )
    if topology.get("logical_cpu_count") not in (None, 64):
        raise ComparisonContractError(f"{system}: expected 64 logical CPUs")
    if topology.get("physical_core_count") != 64:
        raise ComparisonContractError(f"{system}: expected 64 physical cores")
    if topology.get("smt_siblings_selected") is not False:
        raise ComparisonContractError(f"{system}: SMT siblings are selected")
    if topology.get("socket_ids") != EXPECTED_SOCKET_IDS:
        raise ComparisonContractError(
            f"{system}: CPUs span multiple sockets or are not on socket 0"
        )
    if topology.get("numa_node_ids") != EXPECTED_NUMA_NODE_IDS:
        raise ComparisonContractError(f"{system}: NUMA nodes must be exactly 0-3")
    if topology.get("numa_node_core_counts") != EXPECTED_NUMA_NODE_CORE_COUNTS:
        raise ComparisonContractError(
            f"{system}: each of NUMA nodes 0-3 must contribute 16 cores"
        )
    if topology.get("socket_core_counts") != EXPECTED_SOCKET_CORE_COUNTS:
        raise ComparisonContractError(f"{system}: socket 0 must contribute 64 cores")

    numa_policy = summary.get("numa_policy") or {}
    if numa_policy.get("policy") != "interleave":
        raise ComparisonContractError(
            f"{system}: NUMA memory policy must interleave NUMA nodes 0-3"
        )
    expected_policy = {
        "physcpubind": EXPECTED_CPU_IDS,
        "cpubind": EXPECTED_NUMA_NODE_IDS,
        "nodebind": EXPECTED_NUMA_NODE_IDS,
        "interleavemask": EXPECTED_NUMA_NODE_IDS,
    }
    for field, expected in expected_policy.items():
        if _expand_cpu_or_node_set(numa_policy.get(field)) != expected:
            raise ComparisonContractError(
                f"{system}: numactl {field} must be exactly {expected}"
            )

    preflight = summary.get("resource_preflight") or {}
    if preflight.get("clean_required") is not True:
        raise ComparisonContractError(f"{system}: clean preflight was not required")
    if preflight.get("violations") != []:
        raise ComparisonContractError(f"{system}: preflight resource contamination")

    monitor = summary.get("runtime_resource_monitor") or {}
    if monitor.get("continuous") is not True or monitor.get("status") != "clean":
        raise ComparisonContractError(f"{system}: runtime audit is absent or dirty")
    if int(monitor.get("sample_count", 0)) < 2:
        raise ComparisonContractError(f"{system}: insufficient runtime samples")
    if monitor.get("violations") != []:
        raise ComparisonContractError(f"{system}: runtime resource contamination")
    if int(monitor.get("foreign_gpu_process_count_max", -1)) != 0:
        raise ComparisonContractError(f"{system}: foreign GPU process observed")


def _require_metric_contract(summary: Mapping[str, Any]) -> None:
    topk = int(summary["topk"])
    contract = summary.get("metric_contract") or {}
    retrieval = contract.get("retrieval") or {}
    if retrieval.get("name") != "answer_hit_recall_at_k":
        raise ComparisonContractError("retrieval metric is not answer-hit Recall@k")
    if int(retrieval.get("k", -1)) != topk:
        raise ComparisonContractError("retrieval metric k does not match topk")
    if contract.get("generation") != ["f1", "acc"]:
        raise ComparisonContractError("generation metrics must be exactly F1 and ACC")
    metrics = summary.get("metrics") or {}
    for name in (f"retrieval_recall_top{topk}", "f1", "acc"):
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ComparisonContractError(f"missing finite metric: {name}")
    artifact = summary.get("request_metrics_artifact") or {}
    if not artifact.get("sha256"):
        raise ComparisonContractError("request-level metric artifact is missing")
    if int(artifact.get("request_count", -1)) != int(summary["num_requests"]):
        raise ComparisonContractError("request-level metric denominator mismatch")


def _require_latency_accounting(summary: Mapping[str, Any]) -> None:
    system = str(summary.get("system") or "unknown")
    accounting = summary.get("latency_accounting") or {}
    expected = {
        "canonical_origin": "scheduled_arrival",
        "completion": "per_request_final_generation",
        "includes_arrival_schedule_lag": True,
    }
    observed = {name: accounting.get(name) for name in expected}
    if observed != expected:
        raise ComparisonContractError(
            f"{system}: latency accounting must include all queueing: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )


def _require_prompt_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    system = str(summary.get("system"))
    ordering_mode = EXPECTED_ORDERING_MODES[system]
    expected = expected_prompt_contract(
        max_model_len=int(summary["max_model_len"]),
        answer_max_tokens=int(summary["max_tokens"]),
        irg_round_max_tokens=int(summary["max_tokens"]),
        recomp_compression_max_tokens=int(summary["recomp_compression_max_tokens"]),
        ordering_mode=ordering_mode,
    )
    observed = summary.get("prompt_contract")
    if observed != expected:
        raise ComparisonContractError(
            f"{system}: prompt contract mismatch: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )
    return observed


def validate_standalone_point(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one point without asserting a matched cross-system sample."""
    system = str(summary.get("system") or "unknown")
    if system not in REQUIRED_SYSTEMS:
        raise ComparisonContractError(f"unsupported system: {system}")
    if summary.get("status") != "complete":
        raise ComparisonContractError(f"{system}: point is not complete")
    missing = [name for name in COMMON_POINT_FIELDS if name not in summary]
    if missing:
        raise ComparisonContractError(f"{system}: missing fields {missing}")
    _require_clean_resources(summary)
    _require_metric_contract(summary)
    _require_latency_accounting(summary)
    prompt = _require_prompt_contract(summary)
    return {
        "status": "valid",
        "system": system,
        "evaluation_scope": "standalone",
        "cpu_topology": summary["cpu_topology"],
        "numa_policy": summary["numa_policy"],
        "prompt_contract": prompt,
    }


def validate_fair_comparison_group(
    summaries: Sequence[Mapping[str, Any]],
    *,
    required_systems: Sequence[str] = REQUIRED_SYSTEMS,
) -> dict[str, Any]:
    """Validate one point containing one result from each compared system."""
    required_systems = tuple(map(str, required_systems))
    if not required_systems:
        raise ComparisonContractError("required systems must not be empty")
    if len(set(required_systems)) != len(required_systems):
        raise ComparisonContractError("required systems must be unique")
    unsupported = sorted(set(required_systems) - set(REQUIRED_SYSTEMS))
    if unsupported:
        raise ComparisonContractError(f"unsupported systems: {unsupported}")
    if "mnemorag" not in required_systems:
        raise ComparisonContractError("fair comparison must include mnemorag")
    if len(summaries) != len(required_systems):
        raise ComparisonContractError(
            f"expected {len(required_systems)} summaries, observed {len(summaries)}"
        )
    by_system = {str(summary.get("system")): summary for summary in summaries}
    if set(by_system) != set(required_systems):
        raise ComparisonContractError(
            f"systems must be exactly {list(required_systems)}"
        )

    prompt_contracts = {}
    for system, summary in by_system.items():
        if summary.get("status") != "complete":
            raise ComparisonContractError(f"{system}: point is not complete")
        missing = [name for name in COMMON_POINT_FIELDS if name not in summary]
        if missing:
            raise ComparisonContractError(f"{system}: missing fields {missing}")
        _require_clean_resources(summary)
        _require_metric_contract(summary)
        _require_latency_accounting(summary)
        prompt_contracts[system] = _require_prompt_contract(summary)
    positions = sorted(
        int(summary.get("campaign_order_position", -1))
        for summary in by_system.values()
    )
    expected_positions = list(range(len(required_systems)))
    if positions != expected_positions:
        raise ComparisonContractError(
            "campaign order positions must be "
            f"{expected_positions}, observed {positions}"
        )

    reference = by_system[required_systems[0]]
    mismatches: dict[str, dict[str, Any]] = {}
    for field in COMMON_POINT_FIELDS:
        expected = reference[field]
        observed = {
            system: summary[field]
            for system, summary in by_system.items()
            if _canonical(summary[field]) != _canonical(expected)
        }
        if observed:
            mismatches[field] = {
                "expected": expected,
                "observed": observed,
            }

    manifest_keys = ("sample_indices_sha256", "dataset_sha256")
    reference_manifest = reference.get("arrival_trace_manifest") or {}
    for key in manifest_keys:
        expected = reference_manifest.get(key)
        if not expected:
            mismatches[f"arrival_trace_manifest.{key}"] = {
                "expected": "nonempty digest",
                "observed": {"mnemorag": expected},
            }
            continue
        observed = {
            system: (summary.get("arrival_trace_manifest") or {}).get(key)
            for system, summary in by_system.items()
            if (summary.get("arrival_trace_manifest") or {}).get(key) != expected
        }
        if observed:
            mismatches[f"arrival_trace_manifest.{key}"] = {
                "expected": expected,
                "observed": observed,
            }

    reference_topology = reference["cpu_topology"]
    for field in ("logical_cpu_ids", "socket_ids", "numa_node_ids"):
        expected = reference_topology.get(field)
        observed = {
            system: (summary.get("cpu_topology") or {}).get(field)
            for system, summary in by_system.items()
            if (summary.get("cpu_topology") or {}).get(field) != expected
        }
        if observed:
            mismatches[f"cpu_topology.{field}"] = {
                "expected": expected,
                "observed": observed,
            }

    reference_prompt = {
        key: value
        for key, value in prompt_contracts["mnemorag"].items()
        if key not in ORDERING_ONLY_PROMPT_FIELDS
    }
    for system, contract in prompt_contracts.items():
        shared_contract = {
            key: value
            for key, value in contract.items()
            if key not in ORDERING_ONLY_PROMPT_FIELDS
        }
        if shared_contract != reference_prompt:
            mismatches[f"prompt_contract.shared.{system}"] = {
                "expected": reference_prompt,
                "observed": shared_contract,
            }

    if mismatches:
        raise ComparisonContractError(
            "cross-system configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "valid",
        "systems": list(required_systems),
        "common_point": {field: reference[field] for field in COMMON_POINT_FIELDS},
        "sample_indices_sha256": reference_manifest["sample_indices_sha256"],
        "dataset_sha256": reference_manifest["dataset_sha256"],
        "cpu_topology": {
            field: reference_topology.get(field)
            for field in ("logical_cpu_ids", "socket_ids", "numa_node_ids")
        },
        "prompt_contract": {
            "shared": reference_prompt,
            "intentional_ordering_policy": {
                system: {
                    field: prompt_contracts[system][field]
                    for field in ORDERING_ONLY_PROMPT_FIELDS
                }
                for system in required_systems
            },
        },
    }


def validate_quality_noninferiority(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    maximum_absolute_drop: Mapping[str, float],
) -> dict[str, float]:
    candidate_metrics = candidate.get("metrics") or {}
    reference_metrics = reference.get("metrics") or {}
    deltas: dict[str, float] = {}
    failures: dict[str, dict[str, float]] = {}
    for metric, maximum_drop in maximum_absolute_drop.items():
        if maximum_drop < 0:
            raise ValueError("maximum quality drops must be nonnegative")
        delta = float(candidate_metrics[metric]) - float(reference_metrics[metric])
        deltas[metric] = delta
        if delta < -float(maximum_drop):
            failures[metric] = {
                "delta": delta,
                "minimum_allowed_delta": -float(maximum_drop),
            }
    if failures:
        raise ComparisonContractError(
            "quality noninferiority failed: " + json.dumps(failures, sort_keys=True)
        )
    return deltas


def validate_mnemorag_full_optimizations(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Require every mechanism applicable to the selected workflow."""
    if summary.get("system") != "mnemorag":
        raise ComparisonContractError("full-optimization audit requires MnemoRAG")
    workflow = str(summary.get("workflow") or "").lower()
    if workflow not in {"onepass", "irg", "recomp"}:
        raise ComparisonContractError(f"unsupported MnemoRAG workflow: {workflow}")
    onepass = workflow == "onepass"
    required_equal = {
        "components": "full",
        "order_mode": "hotness_hint",
        "enable_similarity_aware": True,
        "enable_preassigned_guidance": True,
        "enable_hotness_sort": True,
        "enable_async_scheduling": True,
        "document_kv_heat": True,
        "cache_aware_prefill_batching": onepass,
        "request_batching_policy": "worker_nonblocking_drain",
        "preformed_work_queue": False,
        "async_retrieval_policy": "eager",
        "async_retrieval_policy_effective": "eager",
        "direct_vllm_dispatch": True,
        "application_generation_gate": False,
        "cross_microbatch_pipeline": False,
        "cross_microbatch_pipeline_effective": False,
        "prefix_caching": True,
    }
    mismatches = {
        name: {"expected": expected, "observed": summary.get(name)}
        for name, expected in required_equal.items()
        if summary.get(name) != expected
    }
    cpu_kv = summary.get("cpu_kv_offload") or {}
    if cpu_kv.get("enabled") is not True:
        mismatches["cpu_kv_offload.enabled"] = {
            "expected": True,
            "observed": cpu_kv.get("enabled"),
        }
    if float(cpu_kv.get("capacity_gib", 0.0)) <= 0:
        mismatches["cpu_kv_offload.capacity_gib"] = {
            "expected": "positive",
            "observed": cpu_kv.get("capacity_gib"),
        }
    if cpu_kv.get("h2d_scheduling") != "async":
        mismatches["cpu_kv_offload.h2d_scheduling"] = {
            "expected": "async",
            "observed": cpu_kv.get("h2d_scheduling"),
        }
    heat_metadata = summary.get("document_heat_metadata_stats") or {}
    request_count = int(summary.get("num_requests") or 0)
    if int(heat_metadata.get("requests") or 0) < request_count:
        mismatches["document_heat_metadata_stats.requests"] = {
            "expected": f">={request_count}",
            "observed": heat_metadata.get("requests"),
        }
    if int(heat_metadata.get("spans") or 0) < request_count:
        mismatches["document_heat_metadata_stats.spans"] = {
            "expected": f">={request_count}",
            "observed": heat_metadata.get("spans"),
        }
    transfer = summary.get("cpu_kv_transfer_latency") or {}
    d2h_admission = transfer.get("d2h_admission") or {}
    observed_requests = int(d2h_admission.get("document_heat_observed_requests") or 0)
    skipped_requests = int(d2h_admission.get("cold_admission_skipped_requests") or 0)
    if observed_requests < request_count:
        mismatches[
            "cpu_kv_transfer_latency.d2h_admission.document_heat_observed_requests"
        ] = {
            "expected": f">={request_count}",
            "observed": observed_requests,
        }
    if float(d2h_admission.get("document_heat_max_seen") or 0.0) < 1.0:
        mismatches["cpu_kv_transfer_latency.d2h_admission.document_heat_max_seen"] = {
            "expected": ">=1.0",
            "observed": d2h_admission.get("document_heat_max_seen"),
        }
    if observed_requests > 0 and skipped_requests >= observed_requests:
        mismatches[
            "cpu_kv_transfer_latency.d2h_admission.cold_admission_skipped_requests"
        ] = {
            "expected": f"<{observed_requests}",
            "observed": skipped_requests,
        }
    d2h_jobs = int(
        (transfer.get("d2h_offload") or {}).get("worker_completed_jobs") or 0
    )
    if d2h_jobs <= 0:
        mismatches["cpu_kv_transfer_latency.d2h_offload.worker_completed_jobs"] = {
            "expected": "positive",
            "observed": d2h_jobs,
        }
    h2d_jobs = int((transfer.get("h2d_reload") or {}).get("worker_completed_jobs") or 0)
    if h2d_jobs <= 0:
        mismatches["cpu_kv_transfer_latency.h2d_reload.worker_completed_jobs"] = {
            "expected": "positive",
            "observed": h2d_jobs,
        }
    prior = summary.get("document_hotness_prior") or {}
    if prior.get("enabled") is not True:
        mismatches["document_hotness_prior.enabled"] = {
            "expected": True,
            "observed": prior.get("enabled"),
        }
    if not isinstance(prior.get("freeze"), bool):
        mismatches["document_hotness_prior.freeze"] = {
            "expected": "explicit boolean",
            "observed": prior.get("freeze"),
        }
    if not prior.get("sha256"):
        mismatches["document_hotness_prior.sha256"] = {
            "expected": "nonempty digest",
            "observed": prior.get("sha256"),
        }
    snapshot = summary.get("query_base_snapshot") or {}
    if snapshot.get("enabled") is not True:
        mismatches["query_base_snapshot.enabled"] = {
            "expected": True,
            "observed": snapshot.get("enabled"),
        }
    if int(snapshot.get("loaded_entries") or 0) <= 0:
        mismatches["query_base_snapshot.loaded_entries"] = {
            "expected": "positive",
            "observed": snapshot.get("loaded_entries"),
        }
    if not snapshot.get("sha256"):
        mismatches["query_base_snapshot.sha256"] = {
            "expected": "nonempty digest",
            "observed": snapshot.get("sha256"),
        }
    scheduler = summary.get("scheduler") or {}
    if scheduler.get("microbatch_policy") != "worker_dispatch_drain":
        mismatches["scheduler.microbatch_policy"] = {
            "expected": "worker_dispatch_drain",
            "observed": scheduler.get("microbatch_policy"),
        }
    if scheduler.get("request_batching_policy") != "worker_nonblocking_drain":
        mismatches["scheduler.request_batching_policy"] = {
            "expected": "worker_nonblocking_drain",
            "observed": scheduler.get("request_batching_policy"),
        }
    if scheduler.get("preformed_work_queue") is not False:
        mismatches["scheduler.preformed_work_queue"] = {
            "expected": False,
            "observed": scheduler.get("preformed_work_queue"),
        }
    for removed_field in ("work_queue_high_water", "microbatch_max_wait_ms"):
        if removed_field in scheduler:
            mismatches[f"scheduler.{removed_field}"] = {
                "expected": "absent",
                "observed": scheduler.get(removed_field),
            }
    retrieval_batcher = scheduler.get("retrieval_batcher") or {}
    if retrieval_batcher.get("policy") != "ready_only_no_timed_window":
        mismatches["scheduler.retrieval_batcher.policy"] = {
            "expected": "ready_only_no_timed_window",
            "observed": retrieval_batcher.get("policy"),
        }
    if float(retrieval_batcher.get("intentional_wait_ms", -1.0)) != 0.0:
        mismatches["scheduler.retrieval_batcher.intentional_wait_ms"] = {
            "expected": 0.0,
            "observed": retrieval_batcher.get("intentional_wait_ms"),
        }
    exact_completion_count = int(
        scheduler.get("exact_request_completion_timestamps") or 0
    )
    fallback_completion_count = int(
        scheduler.get("fallback_batch_completion_timestamps") or 0
    )
    if exact_completion_count != request_count or fallback_completion_count != 0:
        mismatches["scheduler.request_completion_timestamps"] = {
            "expected": {
                "exact": request_count,
                "fallback": 0,
            },
            "observed": {
                "exact": exact_completion_count,
                "fallback": fallback_completion_count,
            },
        }
    stage_scheduler = summary.get("batch_stage_scheduler") or {}
    if onepass and stage_scheduler.get("application_generation_gate") is not False:
        mismatches["batch_stage_scheduler.application_generation_gate"] = {
            "expected": False,
            "observed": stage_scheduler.get("application_generation_gate"),
        }
    if mismatches:
        raise ComparisonContractError(
            "MnemoRAG full-optimization audit failed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "valid",
        "workflow": workflow,
        "mechanisms": required_equal,
        "workflow_specific": {
            "cache_aware_prefill_batching": (
                "enabled" if onepass else "not_applicable"
            ),
            "cross_microbatch_pipeline": (
                "superseded_by_vllm_continuous_batching"
                if onepass
                else "not_applicable"
            ),
            "direct_vllm_dispatch": "enabled",
            "document_heat_cpu_kv": "enabled_and_exercised",
        },
        "cpu_kv_d2h_jobs": d2h_jobs,
        "cpu_kv_h2d_jobs": h2d_jobs,
        "cpu_kv_capacity_gib": float(cpu_kv["capacity_gib"]),
        "document_hotness_freeze_prior": bool(prior["freeze"]),
        "document_hotness_prior_sha256": prior["sha256"],
        "query_base_snapshot_sha256": snapshot["sha256"],
        "query_base_snapshot_entries": int(snapshot["loaded_entries"]),
    }
