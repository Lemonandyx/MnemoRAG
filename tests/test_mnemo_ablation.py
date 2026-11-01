from benchmarks.run_mnemo_ablation import (
    VARIANTS,
    apply_variant,
    variant_environment,
)


def base_command():
    return [
        "python",
        "benchmark_point.py",
        "--components",
        "full",
        "--order-mode",
        "hotness_hint",
        "--async-retrieval-policy",
        "prefill_start",
        "--cpu-kv-cache-gib",
        "238.4",
        "--require-cpu-kv-h2d-samples",
        "--require-cpu-kv-d2h-samples",
    ]


def option(command, name):
    return command[command.index(name) + 1]


def test_no_cpu_kv_removes_sample_requirements_and_capacity():
    command = apply_variant(base_command(), "no_cpu_kv")

    assert option(command, "--cpu-kv-cache-gib") == "0"
    assert "--require-cpu-kv-h2d-samples" not in command
    assert "--require-cpu-kv-d2h-samples" not in command


def test_each_mechanism_has_an_isolated_command_override():
    assert "no_profile_guidance" not in VARIANTS
    assert option(apply_variant(base_command(), "no_sar"), "--components") == "kv_full"
    no_hotness = apply_variant(base_command(), "no_hotness")
    assert option(no_hotness, "--components") == "full_no_hotness"
    assert "--no-document-kv-heat" in no_hotness
    no_async = apply_variant(base_command(), "no_async_pipeline")
    assert option(no_async, "--components") == "full_no_async"
    assert "--no-cross-microbatch-pipeline" in no_async
    assert "--no-cache-aware-prefill-batching" in apply_variant(
        base_command(), "no_cache_aware_prefill"
    )


def test_no_history_removes_both_warmstart_artifacts():
    environment = variant_environment(
        {
            "MNEMORAG_DOC_HOTNESS_PRIOR_PATH": "/prior",
            "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH": "/snapshot",
            "MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR": "1",
        },
        "no_history_warmstart",
    )

    assert "MNEMORAG_DOC_HOTNESS_PRIOR_PATH" not in environment
    assert "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH" not in environment
    assert environment["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] == "0"


def test_profile_environment_is_always_disabled_for_formal_ablation():
    base = {
        "VLLM_KV_PROFILE_NEXT_USE_PATH": "/stale/profile",
        "VLLM_KV_PROFILE_RECORD_PATH": "/stale/record",
        "VLLM_KV_PROFILE_OBSERVE_ONLY": "1",
        "VLLM_KV_ORACLE_NEXT_USE_PATH": "/stale/oracle",
    }

    for variant in VARIANTS:
        environment = variant_environment(base, variant)
        assert "VLLM_KV_PROFILE_NEXT_USE_PATH" not in environment
        assert "VLLM_KV_ORACLE_NEXT_USE_PATH" not in environment
        assert "VLLM_KV_PROFILE_RECORD_PATH" not in environment
        assert "VLLM_KV_PROFILE_OBSERVE_ONLY" not in environment
