#!/usr/bin/env python3
"""Audit six-system retrieval results against the frozen replication contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_CONTRACT = HERE / "six_system_replication_contract.json"
DEFAULT_CAMPAIGN = PROJECT_ROOT / "results/evaluation/retrieval_six_system"
DATASETS = ("squad", "nq", "hotpotqa", "2wikimultihopqa")
SYSTEMS = ("gptcache", "qvcache", "has", "proximity", "hedrarag", "mnemorag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> dict[str, Any]:
    excluded = {".git", "__pycache__", "build", "target", ".pytest_cache"}
    files = []
    digest = hashlib.sha256()
    if not root.exists():
        return {"exists": False, "sha256": None, "files": 0}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        relative = path.relative_to(root)
        file_hash = sha256(path)
        digest.update(str(relative).encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        files.append(str(relative))
    return {"exists": True, "sha256": digest.hexdigest(), "files": len(files)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latency_record(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("system_stats") or {}
    batch_time = stats.get("service_batch_time_mean_s")
    batch_size = stats.get("batch_size_mean")
    service_per_request = None
    if batch_time is not None and batch_size:
        service_per_request = float(batch_time) / float(batch_size)
    service_capacity = (
        1.0 / service_per_request
        if service_per_request is not None and service_per_request > 0
        else None
    )
    offered = float(summary["arrival_rate_qps"])
    overloaded = service_capacity is not None and offered > service_capacity
    return {
        "mean_s": summary.get("latency_mean_s"),
        "p50_s": summary.get("latency_p50_s"),
        "p95_s": summary.get("latency_p95_s"),
        "p99_s": summary.get("latency_p99_s"),
        "offered_qps": offered,
        "service_batch_time_mean_s": batch_time,
        "batch_size_mean": batch_size,
        "estimated_service_capacity_qps": service_capacity,
        "overloaded_by_estimate": overloaded,
    }


def audit_point(
    contract: dict[str, Any], campaign: Path, dataset: str, system: str
) -> dict[str, Any]:
    summary_path = campaign / dataset / system / "summary.json"
    requests_path = campaign / dataset / system / "requests.npz"
    if not summary_path.exists():
        return {
            "dataset": dataset,
            "system": system,
            "status": "missing",
            "headline_eligible": False,
            "failures": ["missing summary.json"],
        }
    summary = load_json(summary_path)
    fair = contract["adapted_fair"]
    failures = []
    checks = {
        "complete": summary.get("status") == "complete",
        "requests": summary.get("num_requests") == fair["requests_per_dataset"],
        "nprobe": summary.get("nprobe") == fair["configured_nprobe"],
        "topk": summary.get("topk") == fair["topk"],
        "qps": math.isclose(
            float(summary.get("arrival_rate_qps", float("nan"))),
            float(fair["arrival_rate_qps"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "cpu_affinity": summary.get("cpu_affinity") == list(range(64)),
        "numa": summary.get("numa_nodes") == [0, 1, 2, 3],
        "requests_npz": requests_path.exists(),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)

    deviations = list(contract["current_v1_deviations"].get(system, []))
    native_eligible = not deviations and bool(
        contract["native_paper"][system].get("headline_eligible") is True
    )
    fair_eligibility = fair["headline_eligibility"][system]
    headline_eligible = bool(fair_eligibility is True and not failures)
    latency = latency_record(summary)
    if latency["overloaded_by_estimate"]:
        headline_eligible = False
        deviations.append("estimated offered load exceeds measured service capacity")

    return {
        "dataset": dataset,
        "system": system,
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "latency": latency,
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "native_paper_eligible": native_eligible,
        "adapted_fair_headline_eligible": headline_eligible,
        "known_deviations": deviations,
        "implementation": summary.get("implementation"),
        "system_stats": summary.get("system_stats"),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Six-System Replication Audit",
        "",
        "This report separates native-paper reproduction from the adapted fair lane.",
        "No current adapted point is promoted to native-paper evidence.",
        "",
        "## Existing Adapted-Fair Points",
        "",
        "| Dataset | System | Mean (ms) | Capacity (QPS) | Overloaded | Fair headline |",
        "|---|---|---:|---:|---|---|",
    ]
    for point in report["points"]:
        latency = point.get("latency") or {}
        mean = latency.get("mean_s")
        capacity = latency.get("estimated_service_capacity_qps")
        lines.append(
            "| {dataset} | {system} | {mean} | {capacity} | {overloaded} | {eligible} |".format(
                dataset=point["dataset"],
                system=point["system"],
                mean=f"{1000.0 * mean:.2f}" if mean is not None else "n/a",
                capacity=f"{capacity:.2f}" if capacity is not None else "n/a",
                overloaded="yes" if latency.get("overloaded_by_estimate") else "no",
                eligible="yes" if point.get("adapted_fair_headline_eligible") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A passing file/configuration check does not make an adapted point a native-paper reproduction.",
            "- Overloaded points remain useful capacity evidence but are excluded from algorithm-only headline bars.",
            "- HedraRAG remains a compatibility backend until an unmodified official-AE run is available.",
            "- The next campaign must record all required stage timings and actual adaptive work.",
            "",
            "## Source Trees",
            "",
            "| System | Exists | Files | Content SHA-256 |",
            "|---|---|---:|---|",
        ]
    )
    for system, evidence in report["source_trees"].items():
        lines.append(
            f"| {system} | {evidence['exists']} | {evidence['files']} | {evidence['sha256'] or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_json(args.contract)
    points = [
        audit_point(contract, args.campaign_root, dataset, system)
        for dataset in DATASETS
        for system in SYSTEMS
    ]
    source_trees = {
        system: tree_digest(Path(os.path.expandvars(root)).resolve())
        for system, root in contract["source_roots"].items()
    }
    report = {
        "schema": "mnemorag.six-system-replication-audit.v1",
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "campaign_root": str(args.campaign_root),
        "points": points,
        "source_trees": source_trees,
        "native_paper_complete": False,
        "adapted_fair_complete": all(point["status"] == "pass" for point in points),
        "goal_complete": False,
        "next_required_artifacts": [
            "native-paper replication outputs for each system with a canonical workload",
            "new adapted-fair campaign with stage and capacity evidence",
            "held-out MnemoRAG optimization gate against unmodified Proximity",
            "three repeated locked evaluation runs and publication figures",
        ],
    }
    json_path = args.output_dir / "replication_audit.json"
    md_path = args.output_dir / "replication_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "points": len(points),
                "native_paper_complete": report["native_paper_complete"],
                "adapted_fair_complete": report["adapted_fair_complete"],
                "output": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
