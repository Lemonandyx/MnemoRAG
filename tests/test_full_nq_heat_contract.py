import hashlib

import numpy as np

import benchmarks.run_full_nq_heat_contract as full_nq_contract
from benchmarks.run_full_nq_heat_contract import (
    REQUESTS,
    figure14_periodic_metrics,
    recovered_capture_manifest,
    validate_formula_summary,
)


def valid_summary(formula="exponential_decay"):
    return {
        "status": "complete",
        "source_queries": REQUESTS,
        "num_requests": REQUESTS,
        "sample_indices": list(range(REQUESTS)),
        "topk": 3,
        "arrival_rate_rps": 24.0,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "vllm_engine_effective": {"max_num_seqs": 64},
        "cpu_kv_offload": {
            "num_cpu_blocks": 130_000,
            "capacity_source": "exact_block_count",
            "kv_cache_dtype": "fp8",
            "block_bytes": 1_048_576,
            "host_memory_mode": "hybrid_pinned_lazy",
            "lazy_allocation": "file_mmap",
            "lazy_mmap_dir": "/data/mnemorag-kv",
            "lazy_backing_required_bytes": 1_048_576,
        },
        "cpu_kv_admission_profile": {
            "enabled": True,
            "mode": "",
            "sha256": "abc",
        },
        "prefix_hit_rate": 0.252,
        "online_prefix_cache_periodic_recent_requests": {
            "metric_definition": (
                "one-second samples of the token-weighted cache-hit rate over "
                "the most recent 1000 completed requests"
            ),
            "sample_interval_s": 1.0,
            "requested_window_size": 1000,
            "sample_count": 3299,
            "mean_hit_rate": 0.225027,
            "max_hit_rate": 0.281,
            "tail_hit_rate": 0.25,
        },
        "kv_cache_stats": {
            "heat_formula": formula,
            "heat_formula_priority_calls": 10,
            "legacy_zero_guard_calls": 0,
            "legacy_time_scale_s": 1.0,
            "profile_priority_mode": "formula_tiebreak",
            "profile_formula_tiebreak_calls": 10,
            "profile_formula_tiebreak_evictions": 2,
            "profile_next_use_accesses": 100,
            "profile_next_use_matches": 100,
            "cpu_kv_transfer": {
                "h2d_reload": {"worker_transferred_blocks": 3},
                "d2h_offload": {"worker_transferred_blocks": 4},
            },
        },
    }


def test_full_nq_contract_accepts_auditable_formula_run():
    result = validate_formula_summary(
        valid_summary(),
        "exponential_decay",
        REQUESTS,
        0.225027,
        0.281,
        0.03,
    )
    assert result["passed"]


def test_full_nq_contract_rejects_profile_formula_bypass():
    summary = valid_summary("legacy_log_denominator")
    summary["kv_cache_stats"]["heat_formula_priority_calls"] = 0
    summary["kv_cache_stats"]["profile_formula_tiebreak_calls"] = 0
    summary["kv_cache_stats"]["profile_formula_tiebreak_evictions"] = 0

    result = validate_formula_summary(
        summary,
        "legacy_log_denominator",
        REQUESTS,
        0.225027,
        0.281,
        0.03,
    )

    assert not result["passed"]
    assert not result["checks"]["formula_executed"]
    assert not result["checks"]["profile_formula_composed"]


def test_cumulative_rate_cannot_substitute_for_figure14_periodic_metric():
    summary = valid_summary()
    summary["prefix_hit_rate"] = 0.303006
    summary["online_prefix_cache_periodic_recent_requests"]["mean_hit_rate"] = 0.1

    result = validate_formula_summary(
        summary,
        "exponential_decay",
        REQUESTS,
        0.225027,
        0.281,
        0.03,
    )

    assert not result["passed"]
    assert not result["checks"]["figure14_average_target"]


def test_legacy_contract_requires_zero_guard_instrumentation():
    summary = valid_summary("legacy_log_denominator")
    del summary["kv_cache_stats"]["legacy_zero_guard_calls"]

    result = validate_formula_summary(
        summary,
        "legacy_log_denominator",
        REQUESTS,
        0.225027,
        0.281,
        0.03,
    )

    assert not result["passed"]
    assert not result["checks"]["legacy_zero_guard_instrumented"]


def test_periodic_metric_reconstruction_verifies_raw_artifact(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "request_metrics.npz"
    np.savez_compressed(
        artifact,
        prefix_cached_tokens=np.asarray([1, 1, 2, 2], dtype="<i8"),
        prefix_prompt_tokens=np.asarray([2, 2, 2, 2], dtype="<i8"),
        finish_offset_s=np.asarray([0.2, 0.4, 1.2, 1.4], dtype="<f8"),
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(full_nq_contract, "REQUESTS", 4)

    metrics = figure14_periodic_metrics(
        {
            "request_metrics_artifact": {
                "path": str(artifact),
                "sha256": digest,
            }
        }
    )

    assert metrics["source"] == "verified_request_metrics_reconstruction"
    assert metrics["sample_count"] == 2
    assert metrics["mean_hit_rate"] == 0.625
    assert metrics["max_hit_rate"] == 0.75
    assert metrics["artifact_sha256_verified"] is True


def test_capture_recovery_requires_completed_workload_and_tail(tmp_path):
    access_log = tmp_path / "access.jsonl"
    access_log.write_text(
        "".join(
            '{"block_hash":"aa","request_position":%d}\n' % position
            for position in range(100)
        )
    )
    run_log = tmp_path / "run.log"
    run_log.write_text("Persistent ONEPASS stats: batches=99, requests=100,\n")

    manifest = recovered_capture_manifest(access_log, run_log, 100)

    assert manifest["status"] == "complete"
    assert manifest["maximum_request_position"] == 99
    assert manifest["request_position_coverage"] == 1.0
