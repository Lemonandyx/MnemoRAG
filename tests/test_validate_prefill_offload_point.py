import json

import pytest

from benchmarks.validate_prefill_offload_point import (
    validate_point,
)


def valid_summary():
    return {
        "status": "complete",
        "async_retrieval_policy": "prefill_start",
        "topk": 5,
        "nprobe": 128,
        "cpu_affinity": list(range(64)),
        "cpu_topology": {
            "physical_core_count": 64,
            "smt_siblings_selected": False,
        },
        "resource_preflight": {"violations": []},
        "runtime_resource_monitor": {
            "status": "clean",
            "continuous": True,
            "sample_count": 3,
            "foreign_gpu_process_count_max": 0,
            "external_cpu_busy_percent_max": 1.0,
        },
        "cpu_kv_offload": {"enabled": True},
        "cpu_kv_transfer_latency": {
            "d2h_offload": {"worker_completed_jobs": 1},
            "h2d_reload": {"worker_completed_jobs": 0},
        },
        "retrieval_ids_sha256": "retrieval",
        "prediction_quality_sha256": "prediction",
        "native_prefill_admission_event": True,
        "request_cycle_size": 0,
        "components": "full",
        "max_tokens": 32,
        "query_base_exact_lookup": True,
    }


def write_summary(tmp_path, row):
    destination = tmp_path / "point" / "summary.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(row), encoding="utf-8")


def test_accepts_complete_runtime_monitored_point(tmp_path):
    write_summary(tmp_path, valid_summary())
    result = validate_point(
        tmp_path, "prefill_start", "unique", ",".join(map(str, range(64)))
    )
    assert result["status"] == "valid"


def test_rejects_contaminated_point(tmp_path):
    row = valid_summary()
    row["runtime_resource_monitor"]["status"] = "contaminated"
    write_summary(tmp_path, row)
    with pytest.raises(RuntimeError, match="runtime_clean"):
        validate_point(
            tmp_path,
            "prefill_start",
            "unique",
            ",".join(map(str, range(64))),
        )
