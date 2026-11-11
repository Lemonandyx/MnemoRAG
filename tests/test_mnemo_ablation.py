from benchmarks.run_mnemo_ablation import (
    PAPER_VARIANTS,
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


def test_each_mechanism_has_an_isolated_command_override():
    assert "no_profile_guidance" not in VARIANTS
    assert option(apply_variant(base_command(), "no_sar"), "--components") == "kv_full"
    no_hotness = apply_variant(base_command(), "no_hotness")
    assert option(no_hotness, "--components") == "full_no_hotness"
    assert "--no-document-kv-heat" in no_hotness
    no_async = apply_variant(base_command(), "no_async_pipeline")
    assert option(no_async, "--components") == "full_no_async"
    assert "--no-cross-microbatch-pipeline" in no_async


def test_paper_ablation_order_and_vanilla_contract():
    assert PAPER_VARIANTS == (
        "vanilla",
        "no_sar",
        "no_hotness",
        "no_async_pipeline",
        "full",
    )
    vanilla = apply_variant(base_command(), "vanilla")
    assert option(vanilla, "--components") == "vanilla"
    assert option(vanilla, "--order-mode") == "original"
    assert option(vanilla, "--async-retrieval-policy") == "sequential"
    assert option(vanilla, "--cpu-kv-cache-gib") == "0"
    assert "--no-document-kv-heat" in vanilla
    assert "--no-cross-microbatch-pipeline" in vanilla
    assert "--no-cache-aware-prefill-batching" in vanilla


def test_vanilla_removes_warmstart_artifacts():
    environment = variant_environment(
        {
            "MNEMORAG_DOC_HOTNESS_PRIOR_PATH": "/prior",
            "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH": "/snapshot",
            "MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR": "1",
        },
        "vanilla",
    )

    assert "MNEMORAG_DOC_HOTNESS_PRIOR_PATH" not in environment
    assert "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH" not in environment
    assert environment["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] == "0"
