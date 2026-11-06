from benchmarks.run_full_nq_heat_contract import (
    REQUESTS,
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
        "prefix_hit_rate": 0.303,
        "kv_cache_stats": {
            "heat_formula": formula,
            "heat_formula_priority_calls": 10,
            "legacy_zero_guard_calls": 1,
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
        valid_summary(), "exponential_decay", REQUESTS, 0.303006, 0.02
    )
    assert result["passed"]


def test_full_nq_contract_rejects_profile_formula_bypass():
    summary = valid_summary("legacy_log_denominator")
    summary["kv_cache_stats"]["heat_formula_priority_calls"] = 0
    summary["kv_cache_stats"]["profile_formula_tiebreak_calls"] = 0
    summary["kv_cache_stats"]["profile_formula_tiebreak_evictions"] = 0

    result = validate_formula_summary(
        summary, "legacy_log_denominator", REQUESTS, 0.303006, 0.02
    )

    assert not result["passed"]
    assert not result["checks"]["formula_executed"]
    assert not result["checks"]["profile_formula_composed"]


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
