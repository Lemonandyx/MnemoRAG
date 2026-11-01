"""Join CPU KV H2D events with request and generation-batch timing."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable


def _overlap_s(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "positive_count": 0,
            "positive_fraction": 0.0,
            "mean_s": 0.0,
            "max_s": 0.0,
            "total_s": 0.0,
        }
    positive_count = sum(value > 1e-9 for value in samples)
    return {
        "count": len(samples),
        "positive_count": positive_count,
        "positive_fraction": positive_count / len(samples),
        "mean_s": statistics.fmean(samples),
        "max_s": max(samples),
        "total_s": sum(samples),
    }


def analyze_cpu_kv_request_trace(
    generation_stage_records: list[dict[str, Any]],
    request_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a causal-ready H2D/Prefill join without changing runtime logic."""
    if not request_trace:
        return {
            "status": "no_h2d_request_trace",
            "trace_count": 0,
            "matched_count": 0,
            "unmatched_request_ids": [],
            "records": [],
        }

    by_request_id: dict[str, dict[str, Any]] = {}
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in generation_stage_records:
        request_id = str(record.get("request_id") or "")
        if not request_id:
            raise RuntimeError("Generation stage record is missing request_id")
        if request_id in by_request_id:
            raise RuntimeError(f"Duplicate generation request_id: {request_id}")
        by_request_id[request_id] = record
        by_batch[int(record["generation_batch_index"])].append(record)

    batch_windows: dict[int, dict[str, Any]] = {}
    for batch_index, records in by_batch.items():
        tail = max(records, key=lambda item: float(item["first_token_s"]))
        batch_windows[batch_index] = {
            "batch_start_s": min(float(item["first_scheduled_s"]) for item in records),
            "batch_end_s": max(float(item["first_token_s"]) for item in records),
            "tail_request_id": str(tail["request_id"]),
        }

    joined_records: list[dict[str, Any]] = []
    unmatched_request_ids: list[str] = []
    scheduled_before_observed = 0
    request_prefill_overlap_count = 0
    for trace in request_trace:
        request_id = str(trace.get("request_id") or "")
        generation = by_request_id.get(request_id)
        if generation is None:
            unmatched_request_ids.append(request_id)
            continue
        batch_index = int(generation["generation_batch_index"])
        batch = batch_windows[batch_index]
        published_s = float(trace["metadata_published_s"])
        worker_started_s = float(trace["worker_started_s"])
        worker_completed_s = float(trace["worker_completed_s"])
        observed_s = float(trace["scheduler_observed_s"])
        scheduled_s = float(generation["first_scheduled_s"])
        first_token_s = float(generation["first_token_s"])
        if scheduled_s + 1e-9 < observed_s:
            scheduled_before_observed += 1
        request_overlap_s = _overlap_s(
            published_s,
            observed_s,
            scheduled_s,
            first_token_s,
        )
        if request_overlap_s > 1e-9:
            request_prefill_overlap_count += 1
        joined_records.append(
            {
                **trace,
                "generation_batch_index": batch_index,
                "batch_position": int(generation["batch_position"]),
                "first_scheduled_s": scheduled_s,
                "first_token_s": first_token_s,
                "prefill_service_s": float(generation["prefill_service_s"]),
                "scheduler_observed_to_scheduled_s": max(0.0, scheduled_s - observed_s),
                "h2d_wait_overlap_with_request_prefill_s": request_overlap_s,
                "h2d_wait_overlap_with_batch_prefill_s": _overlap_s(
                    published_s,
                    observed_s,
                    float(batch["batch_start_s"]),
                    float(batch["batch_end_s"]),
                ),
                "worker_overlap_with_batch_prefill_s": _overlap_s(
                    worker_started_s,
                    worker_completed_s,
                    float(batch["batch_start_s"]),
                    float(batch["batch_end_s"]),
                ),
                "is_tail_request": request_id == batch["tail_request_id"],
            }
        )

    tail_records = [record for record in joined_records if record["is_tail_request"]]
    batch_overlap = [
        float(record["h2d_wait_overlap_with_batch_prefill_s"])
        for record in joined_records
    ]
    worker_batch_overlap = [
        float(record["worker_overlap_with_batch_prefill_s"])
        for record in joined_records
    ]
    return {
        "status": "complete",
        "measurement": (
            "H2D wait ends when the scheduler observes finished_recving. "
            "Request Prefill is first_scheduled to first_token; batch Prefill "
            "is earliest first_scheduled to latest first_token. Intervals are "
            "joined by the native vLLM request_id."
        ),
        "trace_count": len(request_trace),
        "matched_count": len(joined_records),
        "unmatched_request_ids": unmatched_request_ids,
        "scheduled_before_scheduler_observed_count": scheduled_before_observed,
        "request_prefill_overlap_count": request_prefill_overlap_count,
        "h2d_wait_batch_prefill_overlap": _distribution(batch_overlap),
        "worker_batch_prefill_overlap": _distribution(worker_batch_overlap),
        "tail_reload_count": len(tail_records),
        "tail_h2d_wait_batch_prefill_overlap": _distribution(
            float(record["h2d_wait_overlap_with_batch_prefill_s"])
            for record in tail_records
        ),
        "records": joined_records,
    }
