"""Shared validation for resumable, resource-controlled experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

CANONICAL_SYSTEM_PROMPT = (
    "Answer the question based on the given document."
    "Only give me the answer and do not output any other words."
    "\nThe following are given documents.\n\n{reference}"
)
CANONICAL_USER_PROMPT = "Question: {question}"
CANONICAL_DOCUMENT_FORMAT = "Doc {i}(Title: {title}) {text}\\n"
DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS = 128
ORDERING_PROMPT_POLICIES = {
    "retrieval": {
        "document_order": "retrieval_rank",
        "relevance_hint": "none",
    },
    "original": {
        "document_order": "retrieval_rank",
        "relevance_hint": "none",
    },
    "hotness": {
        "document_order": "document_hotness",
        "relevance_hint": "none",
    },
    "hotness_hint": {
        "document_order": "document_hotness",
        "relevance_hint": "original_retrieval_rank_v1",
    },
}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_prompt_contract(
    *,
    max_model_len: int = 2048,
    answer_max_tokens: int = 32,
    irg_round_max_tokens: int = 32,
    recomp_compression_max_tokens: int = DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS,
    ordering_mode: str = "retrieval",
) -> dict:
    """Return the prompt and per-stage generation contract used in comparisons."""
    if ordering_mode not in ORDERING_PROMPT_POLICIES:
        raise ValueError(f"Unsupported prompt ordering mode: {ordering_mode}")
    return {
        "version": 3,
        "template": "flashrag_default_qa_v1",
        "system_prompt_sha256": _text_sha256(CANONICAL_SYSTEM_PROMPT),
        "user_prompt_sha256": _text_sha256(CANONICAL_USER_PROMPT),
        "document_format": CANONICAL_DOCUMENT_FORMAT,
        "ordering_mode": ordering_mode,
        **ORDERING_PROMPT_POLICIES[ordering_mode],
        "truncation": "none",
        "max_model_len": int(max_model_len),
        "onepass_answer_max_tokens": int(answer_max_tokens),
        "irg_round_max_tokens": int(irg_round_max_tokens),
        "recomp_compression_max_tokens": int(recomp_compression_max_tokens),
        "recomp_answer_max_tokens": int(answer_max_tokens),
    }


def verified_complete(
    summary: dict,
    cpu_threads: int,
    required_fields: Iterable[str] = (),
    *,
    expected_point: dict | None = None,
) -> bool:
    topology = summary.get("cpu_topology") or {}
    preflight = summary.get("resource_preflight") or {}
    return bool(
        summary.get("status") == "complete"
        and topology.get("physical_core_count") == cpu_threads
        and not topology.get("smt_siblings_selected", True)
        and preflight.get("clean_required") is True
        and preflight.get("violations") == []
        and all(
            field in summary and summary[field] is not None for field in required_fields
        )
        and verified_point_contract(summary, expected_point or {})
    )


def verified_point_contract(summary: dict, expected: dict) -> bool:
    """Require cached output to match every workload-defining field."""
    missing = object()
    for name, expected_value in expected.items():
        actual_value = summary.get(name, missing)
        if actual_value is missing:
            return False
        if (
            isinstance(expected_value, (int, float))
            and not isinstance(expected_value, bool)
            and isinstance(actual_value, (int, float))
            and not isinstance(actual_value, bool)
        ):
            if not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def verified_quality_contract(summary: dict, topk: int) -> bool:
    """Require exactly Recall@k for retrieval and F1/ACC for generation."""
    contract = summary.get("metric_contract") or {}
    retrieval = contract.get("retrieval") or {}
    metrics = summary.get("metrics") or {}
    observations = summary.get("metric_observations") or {}
    expected_metrics = {f"retrieval_recall_top{int(topk)}", "f1", "acc"}
    num_requests = summary.get("num_requests")
    return bool(
        contract.get("version") == 2
        and retrieval.get("name") == "answer_hit_recall_at_k"
        and retrieval.get("k") == int(topk)
        and retrieval.get("scope") == "final_retrieval_round"
        and retrieval.get("denominator") == "all_requests"
        and contract.get("generation") == ["f1", "acc"]
        and set(metrics) == expected_metrics
        and all(metrics[name] is not None for name in expected_metrics)
        and isinstance(num_requests, int)
        and num_requests > 0
        and set(observations) == expected_metrics
        and all(observations[name] == num_requests for name in expected_metrics)
    )


def verified_prompt_contract(
    summary: dict,
    *,
    max_model_len: int = 2048,
    answer_max_tokens: int = 32,
    irg_round_max_tokens: int = 32,
    recomp_compression_max_tokens: int = DEFAULT_RECOMP_COMPRESSION_MAX_TOKENS,
    ordering_mode: str = "retrieval",
) -> bool:
    """Reject results produced with a different prompt or stage budget."""
    expected = expected_prompt_contract(
        max_model_len=max_model_len,
        answer_max_tokens=answer_max_tokens,
        irg_round_max_tokens=irg_round_max_tokens,
        recomp_compression_max_tokens=recomp_compression_max_tokens,
        ordering_mode=ordering_mode,
    )
    return summary.get("prompt_contract") == expected


def verified_evaluation_contract(
    summary: dict,
    topk: int,
    *,
    ordering_mode: str,
) -> bool:
    """Require both the quality metric and prompt/generation contracts."""
    return verified_quality_contract(summary, topk) and verified_prompt_contract(
        summary, ordering_mode=ordering_mode
    )
