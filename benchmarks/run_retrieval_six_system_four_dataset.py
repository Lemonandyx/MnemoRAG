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

import numpy as np

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
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def history_snapshot(dataset: str) -> Path:
    manifest_path = WORKLOAD_ROOT / "history" / dataset / "history_reuse_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["artifacts"]["query_base_snapshot_path"]["path"])
    expected = manifest["artifacts"]["query_base_snapshot_path"]["sha256"]
    if sha256(path) != expected:
        raise RuntimeError(f"History hash mismatch for {dataset}")
    return path


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


def import_hedra_point(args: argparse.Namespace, dataset: str) -> dict:
    source_dir = (
        WORKLOAD_ROOT
        / "candidate/groups"
        / f"candidate_{dataset}_onepass_k3_np128_qps8_r0"
        / "attempt_001/hedrarag"
        / f"hedrarag_{dataset}_onepass_k3_np128_rps8_n512"
    )
    source_summary_path = source_dir / "summary.json"
    source_metrics_path = source_dir / "request_metrics.npz"
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    expected_trace = json.loads(
        str(np.load(workload_path(dataset), allow_pickle=False)["manifest_json"].item())
    )
    if source.get("status") != "complete":
        raise RuntimeError(f"HedraRAG source is incomplete for {dataset}")
    checks = {
        "system": source.get("system") == "hedrarag",
        "dataset": source.get("dataset") == dataset,
        "num_requests": source.get("num_requests") == 512,
        "nprobe": source.get("nprobe") == 128,
        "topk": source.get("topk") == 3,
        "qps": source.get("arrival_rate_rps") == 8.0,
        "trace": source.get("arrival_trace_sha256")
        == expected_trace["arrival_trace_sha256"],
        "headline_eligible": (source.get("implementation_fidelity") or {}).get(
            "headline_eligible"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"HedraRAG import contract failed: {checks}")

    with np.load(source_metrics_path, allow_pickle=False) as metrics:
        arrival = np.asarray(metrics["arrival_offset_s"], dtype=np.float64)
        finish = np.asarray(metrics["finish_offset_s"], dtype=np.float64)
        ids = np.asarray(metrics["retrieval_ids"], dtype=np.int64)
    point_dir = args.output_dir / dataset / "hedrarag"
    point_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        point_dir / "requests.npz",
        arrival_offset_s=arrival,
        finish_offset_s=finish,
        retrieval_latency_s=np.asarray([], dtype=np.float64),
        returned_doc_ids=ids,
    )
    summary = {
        "status": "complete",
        "system": "hedrarag",
        "dataset": dataset,
        "metric": "request-level retrieval stage measured inside the official AE executor",
        "quality_reference_only": False,
        "num_requests": 512,
        "arrival_rate_qps": 8.0,
        "nprobe": 128,
        "topk": 3,
        "max_batch_size": int(source["retrieval_batch_size"]),
        "batch_policy": "official_ae_continuous_retrieval_executor",
        "physical_gpu_id": int(source["physical_gpu_id"]),
        "cpu_affinity": source["cpu_affinity"],
        "numa_nodes": [0, 1, 2, 3],
        "workload_path": str(workload_path(dataset)),
        "workload_sha256": sha256(workload_path(dataset)),
        "workload_manifest": expected_trace,
        "history_snapshot_path": None,
        "history_snapshot_sha256": None,
        "history_entries": 0,
        "warmup": {
            "history_load_mode": "native_fixed_4GiB_hot_cluster_cache",
            "source_online_warmup": source.get("online_warmup"),
        },
        "system_stats": source.get("retrieval_gpu_cache"),
        "latency_mean_s": float(source["stage_latency"]["retrieval_mean_s"]),
        "latency_p50_s": None,
        "latency_p95_s": float(source["stage_latency"]["retrieval_p95_s"]),
        "latency_p99_s": None,
        "implementation": "official HedraRAG AE heterogeneous CPU/GPU continuous-retrieval executor",
        "headline_eligible": False,
        "source_summary_path": str(source_summary_path),
        "source_summary_sha256": sha256(source_summary_path),
        "source_metrics_path": str(source_metrics_path),
        "source_metrics_sha256": sha256(source_metrics_path),
        "import_contract": checks,
    }
    output = point_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"IMPORT_VALIDATED {dataset} hedrarag", flush=True)
    return summary


def run_point(args: argparse.Namespace, dataset: str, system: str) -> dict:
    summary_path = args.output_dir / dataset / system / "summary.json"
    if point_complete(summary_path, system, dataset):
        print(f"SKIP_COMPLETE {dataset} {system}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if system == "hedrarag":
        return import_hedra_point(args, dataset)

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
        str(history_snapshot(dataset)),
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
                "recall_at_3_vs_faiss": recall,
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
                    "recall_at_3_vs_faiss": recall,
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
            "history_entries": "2047 (SQuAD) or 2048; identical across systems within each dataset",
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
