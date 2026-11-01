"""Planning utilities for the reproducible 10-second SLO campaign."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

SYSTEMS = ("mnemorag", "flashrag", "hedrarag")
WORKFLOWS = ("onepass", "recomp", "irg")
NPROBES = (128, 256, 512)
DATASETS = ("squad", "nq", "hotpotqa", "2wikimultihopqa")
QPS_GRIDS = {
    "onepass": (
        2.0,
        4.0,
        8.0,
        12.0,
        16.0,
        20.0,
        24.0,
        28.0,
        32.0,
        40.0,
        48.0,
        56.0,
        64.0,
    ),
    "recomp": (
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        20.0,
        24.0,
        28.0,
        32.0,
        40.0,
        48.0,
        56.0,
        64.0,
    ),
    "irg": (
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        20.0,
        24.0,
        28.0,
        32.0,
        40.0,
        48.0,
        56.0,
        64.0,
    ),
}


@dataclass(frozen=True)
class PointGroup:
    phase: str
    dataset: str
    workflow: str
    nprobe: int
    qps: float
    repeat: int
    seed: int
    num_requests: int
    system_order: tuple[str, ...]
    topk: int = 3
    latency_slo_seconds: float = 10.0

    def identifier(self) -> str:
        qps = f"{self.qps:g}".replace(".", "p")
        return (
            f"{self.phase}_{self.dataset}_{self.workflow}_k{self.topk}_"
            f"np{self.nprobe}_qps{qps}_r{self.repeat}"
        )

    def records(self) -> list[dict]:
        common = asdict(self)
        common.pop("system_order")
        return [
            common | {"system": system, "order_position": position}
            for position, system in enumerate(self.system_order)
        ]


def reversed_system_order(repeat: int) -> tuple[str, ...]:
    return SYSTEMS if int(repeat) % 2 == 0 else tuple(reversed(SYSTEMS))


def request_count_for_duration(
    qps: float,
    duration_s: float,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if qps <= 0 or duration_s <= 0:
        raise ValueError("qps and duration_s must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid request-count bounds")
    return min(maximum, max(minimum, int(math.ceil(qps * duration_s))))


def candidate_groups(
    *,
    dataset: str = "nq",
    datasets: Sequence[str] | None = None,
    nprobes: Sequence[int] = NPROBES,
    base_seed: int = 20260923,
    duration_s: float = 90.0,
    minimum_requests: int = 128,
    maximum_requests: int = 2048,
) -> list[PointGroup]:
    selected_datasets = (dataset,) if datasets is None else tuple(datasets)
    if not selected_datasets or any(
        selected not in DATASETS for selected in selected_datasets
    ):
        raise ValueError(f"unsupported datasets: {selected_datasets}")
    if not nprobes or any(nprobe not in NPROBES for nprobe in nprobes):
        raise ValueError(f"unsupported nprobes: {nprobes}")
    groups = []
    for selected_dataset in selected_datasets:
        for workflow in WORKFLOWS:
            for nprobe in nprobes:
                for qps in QPS_GRIDS[workflow]:
                    groups.append(
                        PointGroup(
                            phase="candidate",
                            dataset=selected_dataset,
                            workflow=workflow,
                            nprobe=nprobe,
                            qps=qps,
                            repeat=0,
                            seed=base_seed,
                            num_requests=request_count_for_duration(
                                qps,
                                duration_s,
                                minimum=minimum_requests,
                                maximum=maximum_requests,
                            ),
                            system_order=reversed_system_order(len(groups)),
                        )
                    )
    return groups


def _passes_slo(
    summary: Mapping[str, object],
    *,
    minimum_attainment: float,
    latency_slo_seconds: float,
) -> bool:
    return bool(
        float(summary["slo_attainment"]) >= minimum_attainment
        and float(summary["request_latency_p99_s"]) <= latency_slo_seconds
    )


def select_boundary_qps(
    candidate_summaries: Iterable[Mapping[str, object]],
    *,
    minimum_attainment: float = 0.99,
    latency_slo_seconds: float = 10.0,
    nprobes: Sequence[int] = NPROBES,
) -> dict[tuple[str, int], tuple[float, ...]]:
    """Select a shared bracket around every system's observed SLO boundary."""
    if not 0 < minimum_attainment <= 1:
        raise ValueError("minimum_attainment must be in (0, 1]")
    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for summary in candidate_summaries:
        key = (
            str(summary["workflow"]),
            int(summary["nprobe"]),
            str(summary["system"]),
        )
        grouped[key].append(summary)

    selected: dict[tuple[str, int], set[float]] = defaultdict(set)
    expected_keys = {
        (workflow, nprobe, system)
        for workflow in WORKFLOWS
        for nprobe in nprobes
        for system in SYSTEMS
    }
    missing = sorted(expected_keys - set(grouped))
    if missing:
        raise ValueError(f"candidate scan is incomplete: {missing}")

    for (workflow, nprobe, system), rows in grouped.items():
        by_qps = {float(row["arrival_rate_rps"]): row for row in rows}
        expected_qps = tuple(map(float, QPS_GRIDS[workflow]))
        if tuple(sorted(by_qps)) != expected_qps:
            raise ValueError(
                f"candidate QPS grid mismatch for {(workflow, nprobe, system)}"
            )
        passing = [
            index
            for index, qps in enumerate(expected_qps)
            if _passes_slo(
                by_qps[qps],
                minimum_attainment=minimum_attainment,
                latency_slo_seconds=latency_slo_seconds,
            )
        ]
        if not passing:
            indices = range(min(2, len(expected_qps)))
        elif passing[-1] == len(expected_qps) - 1:
            indices = range(max(0, len(expected_qps) - 2), len(expected_qps))
        else:
            boundary = passing[-1]
            indices = range(max(0, boundary - 1), min(len(expected_qps), boundary + 2))
        selected[(workflow, nprobe)].update(expected_qps[index] for index in indices)
    return {key: tuple(sorted(values)) for key, values in sorted(selected.items())}


def select_boundary_qps_by_dataset(
    candidate_summaries: Iterable[Mapping[str, object]],
    *,
    datasets: Sequence[str] = DATASETS,
    minimum_attainment: float = 0.99,
    latency_slo_seconds: float = 10.0,
    nprobes: Sequence[int] = NPROBES,
) -> dict[tuple[str, str, int], tuple[float, ...]]:
    rows = list(candidate_summaries)
    selected: dict[tuple[str, str, int], tuple[float, ...]] = {}
    for dataset in datasets:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == dataset]
        if not dataset_rows:
            raise ValueError(f"candidate scan is missing dataset: {dataset}")
        for (workflow, nprobe), qps_values in select_boundary_qps(
            dataset_rows,
            minimum_attainment=minimum_attainment,
            latency_slo_seconds=latency_slo_seconds,
            nprobes=nprobes,
        ).items():
            selected[(dataset, workflow, nprobe)] = qps_values
    return selected


def final_slo_groups(
    selected_qps: Mapping[tuple[str, int], Sequence[float]],
    *,
    dataset: str = "nq",
    nprobes: Sequence[int] = NPROBES,
    repeats: int = 2,
    base_seed: int = 20260923,
    duration_s: float = 300.0,
    minimum_requests: int = 512,
    maximum_requests: int = 4096,
) -> list[PointGroup]:
    if repeats <= 0:
        raise ValueError("formal SLO points require at least one measurement")
    groups = []
    for workflow in WORKFLOWS:
        for nprobe in nprobes:
            key = (workflow, nprobe)
            qps_values = tuple(selected_qps.get(key, ()))
            if not qps_values:
                raise ValueError(f"missing selected QPS for {key}")
            for qps in qps_values:
                for repeat in range(repeats):
                    groups.append(
                        PointGroup(
                            phase="final_slo",
                            dataset=dataset,
                            workflow=workflow,
                            nprobe=nprobe,
                            qps=float(qps),
                            repeat=repeat,
                            seed=base_seed + repeat,
                            num_requests=request_count_for_duration(
                                float(qps),
                                duration_s,
                                minimum=minimum_requests,
                                maximum=maximum_requests,
                            ),
                            system_order=reversed_system_order(repeat),
                        )
                    )
    return groups


def final_slo_groups_by_dataset(
    selected_qps: Mapping[tuple[str, str, int], Sequence[float]],
    *,
    datasets: Sequence[str] = DATASETS,
    nprobes: Sequence[int] = NPROBES,
    repeats: int = 2,
    base_seed: int = 20260923,
    duration_s: float = 300.0,
    minimum_requests: int = 512,
    maximum_requests: int = 4096,
) -> list[PointGroup]:
    groups = []
    for dataset in datasets:
        per_dataset = {
            (workflow, nprobe): selected_qps[(dataset, workflow, nprobe)]
            for workflow in WORKFLOWS
            for nprobe in nprobes
        }
        groups.extend(
            final_slo_groups(
                per_dataset,
                dataset=dataset,
                nprobes=nprobes,
                repeats=repeats,
                base_seed=base_seed,
                duration_s=duration_s,
                minimum_requests=minimum_requests,
                maximum_requests=maximum_requests,
            )
        )
    return groups


def mnemo_full_groups(
    selected_qps: Mapping[tuple[str, str, int], Sequence[float]],
    request_counts: Mapping[str, int],
    *,
    datasets: Sequence[str] = DATASETS,
    base_seed: int = 20260923,
) -> list[PointGroup]:
    """Build current-semantics MnemoRAG runs over every available eval row."""
    groups = []
    for nprobe in NPROBES:
        for dataset in datasets:
            request_count = int(request_counts.get(dataset, 0))
            if request_count <= 0:
                raise ValueError(f"missing full-split request count for {dataset}")
            for workflow in WORKFLOWS:
                key = (dataset, workflow, nprobe)
                qps_values = tuple(selected_qps.get(key, ()))
                if not qps_values:
                    raise ValueError(f"missing selected QPS for {key}")
                for qps in qps_values:
                    groups.append(
                        PointGroup(
                            phase="mnemo_full",
                            dataset=dataset,
                            workflow=workflow,
                            nprobe=nprobe,
                            qps=float(qps),
                            repeat=0,
                            seed=base_seed,
                            num_requests=request_count,
                            system_order=("mnemorag",),
                        )
                    )
    return groups


def dataset_latency_groups(
    *,
    qps: float,
    nprobe: int | None = None,
    nprobes: Sequence[int] | None = None,
    workflow: str | None = None,
    repeats: int = 2,
    num_requests: int = 1024,
    base_seed: int = 20260923,
) -> list[PointGroup]:
    workflows = WORKFLOWS if workflow is None else (workflow,)
    if nprobe is not None and nprobes is not None:
        raise ValueError("use either nprobe or nprobes, not both")
    selected_nprobes = (
        tuple(nprobes)
        if nprobes is not None
        else (NPROBES if nprobe is None else (nprobe,))
    )
    if any(value not in WORKFLOWS for value in workflows) or any(
        value not in NPROBES for value in selected_nprobes
    ):
        raise ValueError("unsupported workflow or nprobe")
    groups = []
    for dataset in DATASETS:
        for selected_workflow in workflows:
            for selected_nprobe in selected_nprobes:
                for repeat in range(repeats):
                    groups.append(
                        PointGroup(
                            phase="dataset_latency",
                            dataset=dataset,
                            workflow=selected_workflow,
                            nprobe=selected_nprobe,
                            qps=float(qps),
                            repeat=repeat,
                            seed=base_seed + repeat,
                            num_requests=num_requests,
                            system_order=reversed_system_order(repeat),
                        )
                    )
    return groups
