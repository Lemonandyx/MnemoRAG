"""Repeat-aware statistics for the formal cross-system campaign."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

try:
    from .comparison_validation import (
        REQUIRED_SYSTEMS,
        ComparisonContractError,
        validate_fair_comparison_group,
        validate_standalone_point,
    )
except ImportError:
    from comparison_validation import (
        REQUIRED_SYSTEMS,
        ComparisonContractError,
        validate_fair_comparison_group,
        validate_standalone_point,
    )


SUMMARY_METRICS = (
    "slo_attainment",
    "slo_goodput_qps",
    "request_latency_mean_s",
    "request_latency_p50_s",
    "request_latency_p95_s",
    "request_latency_p99_s",
    "throughput_qps",
)

# Two-sided 95% Student-t critical values. Formal campaigns use two or more
# repeat-level observations, so the table avoids a heavyweight SciPy runtime
# dependency while preserving the exact small-sample interval definition.
T95_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t95_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom <= 30:
        return T95_CRITICAL[degrees_of_freedom]
    if degrees_of_freedom <= 40:
        return 2.021
    if degrees_of_freedom <= 60:
        return 2.000
    if degrees_of_freedom <= 120:
        return 1.980
    return 1.960


def mean_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    *,
    allow_single: bool = False,
) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite:
        raise ValueError("at least one observation is required")
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("confidence interval values must be finite")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("only the campaign's required 95% CI is supported")
    mean = statistics.fmean(finite)
    if len(finite) == 1:
        if not allow_single:
            raise ValueError(
                "at least two repeats are required for a confidence interval"
            )
        return {
            "n": 1,
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "ci_half_width": 0.0,
            "confidence": None,
            "ci_available": False,
            "interval_basis": "single_measurement_point_estimate",
        }
    standard_error = statistics.stdev(finite) / math.sqrt(len(finite))
    critical = _t95_critical(len(finite) - 1)
    half_width = critical * standard_error
    return {
        "n": len(finite),
        "mean": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_half_width": half_width,
        "confidence": confidence,
        "ci_available": True,
        "interval_basis": "student_t_repeat_level",
    }


def _base_point_key(summary: Mapping[str, Any]) -> tuple:
    return (
        str(summary["dataset"]),
        str(summary["workflow"]),
        int(summary["topk"]),
        int(summary["nprobe"]),
        float(summary["arrival_rate_rps"]),
    )


def validate_repeated_groups(
    summaries: Iterable[Mapping[str, Any]],
    *,
    minimum_repeats: int = 2,
    allow_incomplete_system_groups: bool = False,
) -> dict[tuple, dict[int, list[Mapping[str, Any]]]]:
    if minimum_repeats < 1:
        raise ValueError("minimum_repeats must be positive")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_group[str(summary["campaign_group_id"])].append(summary)
    if not by_group:
        raise ComparisonContractError("campaign contains no groups")

    by_point: dict[tuple, dict[int, list[Mapping[str, Any]]]] = defaultdict(dict)
    for group_id, group in by_group.items():
        observed_systems = [str(summary["system"]) for summary in group]
        complete = set(observed_systems) == set(REQUIRED_SYSTEMS)
        if complete:
            validate_fair_comparison_group(group)
        elif allow_incomplete_system_groups:
            if not observed_systems or len(observed_systems) != len(
                set(observed_systems)
            ):
                raise ComparisonContractError(
                    f"{group_id}: invalid SLO-censored system set {observed_systems}"
                )
            if not set(observed_systems).issubset(set(REQUIRED_SYSTEMS)):
                raise ComparisonContractError(
                    f"{group_id}: unknown systems {observed_systems}"
                )
            for summary in group:
                validate_standalone_point(summary)
        else:
            validate_fair_comparison_group(group)
        repeats = {int(summary["campaign_repeat"]) for summary in group}
        if len(repeats) != 1:
            raise ComparisonContractError(f"{group_id}: mixed repeat identifiers")
        repeat = repeats.pop()
        point_key = _base_point_key(group[0])
        if repeat in by_point[point_key]:
            raise ComparisonContractError(
                f"duplicate repeat {repeat} for point {point_key}"
            )
        by_point[point_key][repeat] = group

    for point_key, repeats in by_point.items():
        if len(repeats) < minimum_repeats:
            raise ComparisonContractError(
                f"point {point_key} has {len(repeats)} repeats; "
                f"expected at least {minimum_repeats}"
            )
        for repeat, group in repeats.items():
            observed_order = tuple(
                summary["system"]
                for summary in sorted(
                    group, key=lambda item: int(item["campaign_order_position"])
                )
            )
            if allow_incomplete_system_groups:
                positions = [
                    int(summary["campaign_order_position"]) for summary in group
                ]
                if len(positions) != len(set(positions)) or any(
                    position < 0 or position >= len(REQUIRED_SYSTEMS)
                    for position in positions
                ):
                    raise ComparisonContractError(
                        f"point {point_key} repeat {repeat}: invalid order positions "
                        f"{positions}"
                    )
                continue
            expected_order = (
                REQUIRED_SYSTEMS
                if repeat % 2 == 0
                else tuple(reversed(REQUIRED_SYSTEMS))
            )
            if observed_order != expected_order:
                raise ComparisonContractError(
                    f"point {point_key} repeat {repeat}: expected order "
                    f"{expected_order}, observed {observed_order}"
                )
    return by_point


def aggregate_repeated_groups(
    summaries: Iterable[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
    minimum_repeats: int = 2,
    allow_incomplete_system_groups: bool = False,
) -> list[dict[str, Any]]:
    materialized = list(summaries)
    by_point = validate_repeated_groups(
        materialized,
        minimum_repeats=minimum_repeats,
        allow_incomplete_system_groups=allow_incomplete_system_groups,
    )
    output = []
    for point_key, repeats in sorted(by_point.items()):
        dataset, workflow, topk, nprobe, qps = point_key
        observed_systems = tuple(
            system
            for system in REQUIRED_SYSTEMS
            if any(
                row["system"] == system for group in repeats.values() for row in group
            )
        )
        by_system = {}
        for system in observed_systems:
            rows = [
                row
                for repeat in sorted(repeats)
                for row in repeats[repeat]
                if row["system"] == system
            ]
            if len(rows) < minimum_repeats:
                raise ComparisonContractError(
                    f"point {point_key} system {system} has {len(rows)} repeats; "
                    f"expected at least {minimum_repeats}"
                )
            by_system[system] = rows
        for system, rows in by_system.items():
            metric_names = list(SUMMARY_METRICS)
            quality_names = sorted((rows[0].get("metrics") or {}).keys())
            metric_stats = {
                name: mean_confidence_interval(
                    [float(row[name]) for row in rows],
                    confidence,
                    allow_single=len(rows) == 1,
                )
                for name in metric_names
            }
            quality_stats = {
                name: mean_confidence_interval(
                    [float((row.get("metrics") or {})[name]) for row in rows],
                    confidence,
                    allow_single=len(rows) == 1,
                )
                for name in quality_names
            }
            output.append(
                {
                    "dataset": dataset,
                    "workflow": workflow,
                    "topk": topk,
                    "nprobe": nprobe,
                    "arrival_rate_rps": qps,
                    "system": system,
                    "repeat_count": len(rows),
                    "metrics": metric_stats,
                    "quality": quality_stats,
                }
            )
    return output


def robust_slo_pass(
    aggregated_row: Mapping[str, Any],
    *,
    minimum_attainment: float = 0.99,
    latency_slo_seconds: float = 10.0,
) -> bool:
    metrics = aggregated_row["metrics"]
    return bool(
        float(metrics["slo_attainment"]["ci_low"]) >= minimum_attainment
        and float(metrics["request_latency_p99_s"]["ci_high"]) <= latency_slo_seconds
    )
