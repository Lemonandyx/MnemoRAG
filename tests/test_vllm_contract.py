from types import SimpleNamespace

import pytest

from benchmarks.vllm_contract import (
    validate_effective_vllm_contract,
)


def fake_model():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=64,
                max_num_batched_tokens=12288,
            ),
            model_config=SimpleNamespace(max_model_len=2048),
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.55,
                enable_prefix_caching=True,
            ),
        )
    )


def test_effective_contract_reads_nested_vllm_config():
    audit = validate_effective_vllm_contract(
        fake_model(),
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 12288,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.55,
            "enable_prefix_caching": True,
        },
    )

    assert audit["status"] == "validated"
    assert audit["values"]["max_num_batched_tokens"] == 12288


def test_effective_contract_rejects_requested_only_configuration():
    with pytest.raises(RuntimeError, match="missing"):
        validate_effective_vllm_contract(
            SimpleNamespace(),
            {"max_num_seqs": 64},
        )


def test_effective_contract_rejects_runtime_mismatch():
    with pytest.raises(RuntimeError, match="mismatches"):
        validate_effective_vllm_contract(
            fake_model(),
            {"max_num_seqs": 128},
        )
