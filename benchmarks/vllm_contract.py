"""Read and validate the effective vLLM scheduler/cache configuration."""

from __future__ import annotations

import math

FIELDS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "kv_cache_dtype",
)
CHILDREN = (
    "vllm_config",
    "engine_config",
    "scheduler_config",
    "model_config",
    "cache_config",
    "engine",
    "engine_core",
    "llm_engine",
)
FIELD_ALIASES = {
    "kv_cache_dtype": ("kv_cache_dtype", "cache_dtype"),
}


def _configuration_objects(root, maximum_depth=4):
    queue = [(root, "model", 0)]
    seen = set()
    while queue:
        value, path, depth = queue.pop(0)
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        yield value, path
        if depth >= maximum_depth:
            continue
        for name in CHILDREN:
            try:
                child = getattr(value, name, None)
            except Exception:
                child = None
            if child is not None:
                queue.append((child, f"{path}.{name}", depth + 1))


def read_effective_vllm_contract(model) -> dict:
    observed = {}
    sources = {}
    objects = list(_configuration_objects(model))
    for field in FIELDS:
        for value, path in objects:
            for attribute in FIELD_ALIASES.get(field, (field,)):
                try:
                    candidate = getattr(value, attribute, None)
                except Exception:
                    candidate = None
                if candidate is None:
                    continue
                if field in {"max_num_seqs", "max_num_batched_tokens", "max_model_len"}:
                    candidate = int(candidate)
                elif field == "gpu_memory_utilization":
                    candidate = float(candidate)
                elif field == "enable_prefix_caching":
                    candidate = bool(candidate)
                else:
                    candidate = str(candidate)
                    if "float8" in candidate or candidate.startswith("fp8_"):
                        candidate = "fp8"
                observed[field] = candidate
                sources[field] = f"{path}.{attribute}"
                break
            if field in observed:
                break
    return {"values": observed, "sources": sources}


def validate_effective_vllm_contract(model, expected: dict) -> dict:
    audit = read_effective_vllm_contract(model)
    values = audit["values"]
    missing = [field for field in expected if field not in values]
    mismatches = {}
    for field, wanted in expected.items():
        if field not in values:
            continue
        actual = values[field]
        equal = (
            math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9)
            if isinstance(wanted, float)
            else actual == wanted
        )
        if not equal:
            mismatches[field] = {"expected": wanted, "observed": actual}
    if missing or mismatches:
        raise RuntimeError(
            "effective vLLM configuration mismatch: "
            f"missing={missing}, mismatches={mismatches}, sources={audit['sources']}"
        )
    return {
        "status": "validated",
        "values": {field: values[field] for field in expected},
        "sources": {field: audit["sources"][field] for field in expected},
    }
