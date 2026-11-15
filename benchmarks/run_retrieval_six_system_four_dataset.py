#!/usr/bin/env python3
"""Run and aggregate the 4-dataset, 6-system nprobe=128 retrieval comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from .run_full_e2e_campaign import load_profile, load_history_indices
except ImportError:
    from run_full_e2e_campaign import load_profile, load_history_indices

import numpy as np

try:
    from .fair_workload import file_sha256 as sha256
except ImportError:
    from fair_workload import file_sha256 as sha256

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
WORKLOAD_ROOT = Path(
    os.environ.get(
        "MNEMORAG_WORKLOAD_ROOT",
        PROJECT_ROOT / "results/evaluation/end_to_end",
    )
).resolve()
DATASETS = ("squad", "nq", "hotpotqa", "2wikimultihopqa")
SYSTEMS = ("gptcache", "qvcache", "has", "proximity", "hedrarag", "mnemorag")
GPU_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation/retrieval_six_system",
    )
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument("--mnemo-profile", type=Path, required=True)
    return parser.parse_args()


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def gpu_state() -> dict:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={GPU_ID}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = [float(value.strip()) for value in output.split(",")]
    return {"memory_used_mib": memory, "utilization_percent": utilization}


def wait_for_gpu(skip: bool) -> dict:
    while True:
        state = gpu_state()
        if skip or (
            state["memory_used_mib"] <= 256 and state["utilization_percent"] <= 10
        ):
            return state
        print(f"WAIT_GPU1 {json.dumps(state, sort_keys=True)}", flush=True)
        time.sleep(15)


def workload_path(dataset: str) -> Path:
    return (
        WORKLOAD_ROOT
        / "candidate/groups"
        / f"candidate_{dataset}_onepass_k3_np128_qps8_r0/workload.npz"
    )


def history_snapshot(profile_path: Path, dataset: str) -> Path:
    _, _, profile = load_profile(profile_path, dataset)
    _, history = load_history_indices(profile, dataset)
    return Path(history["query_base_snapshot_path"])


def point_complete(path: Path, system: str, dataset: str) -> bool:
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(
        (
            summary.get("status") == "complete",
            summary.get("system") == system,
            summary.get("dataset") == dataset,
            summary.get("num_requests") == 512,
            summary.get("nprobe") == 128,
            summary.get("topk") == 3,
            summary.get("arrival_rate_qps") is not None
            and abs(float(summary["arrival_rate_qps"]) - 8.0) < 1e-9,
        )
    )


def run_point(args: argparse.Namespace, dataset: str, system: str) -> dict:
    summary_path = args.output_dir / dataset / system / "summary.json"
    if point_complete(summary_path, system, dataset):
        print(f"SKIP_COMPLETE {dataset} {system}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    preflight = wait_for_gpu(args.skip_gpu_wait)
    command = [
        "numactl",
        "--physcpubind=0-63",
        "--interleave=0-3",
        sys.executable,
        str(HERE / "retrieval_six_system_point.py"),
        "--system",
        system,
        "--dataset",
        dataset,
        "--workload",
        str(workload_path(dataset)),
        "--history-snapshot",
        str(history_snapshot(args.mnemo_profile, dataset)),
        "--output-dir",
        str(args.output_dir),
        "--nprobe",
        "128",
        "--topk",
        "3",
        "--gpu-id",
        str(GPU_ID),
        "--cpu-threads",
        "64",
        "--max-batch-size",
        "64",
    ]
    if system == "faiss_reference":
        command.append("--quality-reference-only")
    print(
        f"RUN {dataset} {system} preflight={json.dumps(preflight, sort_keys=True)}",
        flush=True,
    )
    point_log = args.output_dir / "logs" / f"{dataset}_{system}.log"
    point_log.parent.mkdir(parents=True, exist_ok=True)
    with point_log.open("a", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    if not point_complete(summary_path, system, dataset):
        raise RuntimeError(f"Point did not validate: {dataset}/{system}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def aggregate(args: argparse.Namespace, start_hash: str) -> None:
    rows = []
    quality = []
    for dataset in DATASETS:
        reference_path = args.output_dir / dataset / "faiss_reference" / "requests.npz"
        reference = np.load(reference_path)["returned_doc_ids"]
        for system in SYSTEMS:
            summary_path = args.output_dir / dataset / system / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            candidate = np.load(args.output_dir / dataset / system / "requests.npz")[
                "returned_doc_ids"
            ]
            if candidate.shape != reference.shape or candidate.shape != (512, 3):
                raise ValueError(
                    f"{dataset}/{system}: result shape differs from the exact reference"
                )
            if np.any(candidate < 0) or np.any(reference < 0):
                raise ValueError(f"{dataset}/{system}: missing retrieval results")
            recall = float(
                np.mean(
                    [
                        len(set(row.tolist()) & set(target.tolist())) / 3.0
                        for row, target in zip(candidate, reference)
                    ]
                )
            )
            exact = float(np.mean(np.all(candidate == reference, axis=1)))
            record = {
                "dataset": dataset,
                "system": system,
                "nprobe": 128,
                "topk": 3,
                "qps": 8.0,
                "requests": 512,
                "mean_retrieval_latency_ms": 1000.0 * summary["latency_mean_s"],
                "p50_retrieval_latency_ms": (
                    1000.0 * summary["latency_p50_s"]
                    if summary["latency_p50_s"] is not None
                    else None
                ),
                "p95_retrieval_latency_ms": 1000.0 * summary["latency_p95_s"],
                "recall_at_3_vs_exact": recall,
                "exact_result_rate": exact,
                "history_entries": summary["history_entries"],
                "implementation": summary["implementation"],
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
            }
            rows.append(record)
            quality.append(
                {
                    "dataset": dataset,
                    "system": system,
                    "recall_at_3_vs_exact": recall,
                    "exact_result_rate": exact,
                }
            )

    csv_path = args.output_dir / "retrieval_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "retrieval_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "quality_audit.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )

    end_hash = source_digest(
        [HERE / "retrieval_six_system_point.py", Path(__file__).resolve()]
    )
    campaign = {
        "status": "complete",
        "datasets": list(DATASETS),
        "systems": list(SYSTEMS),
        "groups_complete": len(rows),
        "groups_expected": len(DATASETS) * len(SYSTEMS),
        "measurement_contract": {
            "nprobe": 128,
            "topk": 3,
            "qps": 8,
            "requests": 512,
            "history_entries": 256,
            "gpu": "physical GPU1",
            "cpu": "socket0 CPU0-63, 64 physical cores, NUMA0-3",
        },
        "source_start_sha256": start_hash,
        "source_end_sha256": end_hash,
        "source_hash_match": start_hash == end_hash,
        "retrieval_comparison_csv_sha256": sha256(csv_path),
    }
    if not campaign["source_hash_match"]:
        raise RuntimeError("Source changed during retrieval campaign")
    (args.output_dir / "campaign_complete.json").write_text(
        json.dumps(campaign, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [HERE / "retrieval_six_system_point.py", Path(__file__).resolve()]
    start_hash = source_digest(sources)
    provenance = {
        "status": "started",
        "source_sha256": start_hash,
        "sources": {str(path): sha256(path) for path in sources},
        "started_at_unix_s": time.time(),
    }
    (args.output_dir / "source_provenance_start.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    for dataset in DATASETS:
        run_point(args, dataset, "faiss_reference")
        for system in SYSTEMS:
            run_point(args, dataset, system)
    aggregate(args, start_hash)
    print(f"CAMPAIGN_COMPLETE {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
