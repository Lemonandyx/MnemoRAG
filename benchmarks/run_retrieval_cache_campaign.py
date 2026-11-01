#!/usr/bin/env python3
"""Run the complete 5-system retrieval-only scaling matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from .experiment_validation import verified_complete
except ImportError:
    from experiment_validation import verified_complete


HERE = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = HERE.parent
PROJECT_ROOT = str(PROJECT_ROOT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument(
        "--workload-mode",
        choices=("iid_unique", "exact_repeat", "mixed_repeat"),
        default="iid_unique",
    )
    parser.add_argument("--unique-queries", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("mnemorag", "faiss", "proximity", "qvcache", "has"),
        default=["mnemorag", "faiss", "proximity", "qvcache", "has"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT_PATH / "results/evaluation/retrieval_cache_scaling",
    )
    args = parser.parse_args()
    if "faiss" not in args.systems:
        parser.error("--systems must include faiss as the quality reference")
    if (
        args.workload_mode == "iid_unique"
        and args.unique_queries is not None
        and args.unique_queries != args.num_requests
    ):
        parser.error("iid_unique requires --unique-queries to equal --num-requests")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = [
        (system, nprobe, rate)
        for nprobe in (128, 256, 512)
        for system in args.systems
        for rate in (4, 8, 12, 16, 20, 24, 28)
    ]
    summaries = []
    for number, (system, nprobe, rate) in enumerate(points, 1):
        point_name = (
            f"{system}_np{nprobe}_rps{rate}_k{args.topk}_"
            f"n{args.num_requests}_{args.workload_mode}_nqtest"
        )
        summary_path = args.output_dir / point_name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            repeat_unique = args.unique_queries or max(1, args.num_requests // 4)
            if args.workload_mode == "iid_unique":
                expected_unique = args.num_requests
            elif args.workload_mode == "exact_repeat":
                expected_unique = repeat_unique
            else:
                expected_unique = (
                    repeat_unique + (args.num_requests - repeat_unique) // 2
                )
            expected_point = {
                "system": system,
                "dataset": "nq_test",
                "seed": 20260926,
                "num_requests": args.num_requests,
                "workload_mode": args.workload_mode,
                "unique_queries": expected_unique,
                "arrival_rate_rps": rate,
                "nprobe": nprobe,
                "topk": args.topk,
                "retrieval_batch_size": 256,
                "batch_wait_ms": 100.0,
                "direct_faiss_mode": "query_parallel",
                "faiss_omp_threads": 1,
                "faiss_search_workers": 64,
                "cache_capacity": 80_000,
                "similarity_t_reuse": 0.95 if system == "mnemorag" else None,
                "similarity_t_guided": 0.70 if system == "mnemorag" else None,
                "exact_query_cache": True if system == "mnemorag" else None,
                "proximity_tau": 0.48 if system == "proximity" else None,
                "has_threshold": 0.6 if system == "has" else None,
                "has_cache_capacity": 2_048 if system == "has" else None,
                "qvcache_regional_theta": True if system == "qvcache" else None,
                "qvcache_lazy_theta_updates": (True if system == "qvcache" else None),
                "qvcache_metric": "cosine" if system == "qvcache" else None,
            }
            if verified_complete(
                summary,
                args.cpu_threads,
                expected_point=expected_point,
            ):
                print(f"SKIP_COMPLETE {number}/{len(points)} {point_name}", flush=True)
                summaries.append(summary)
                continue
            print(
                f"RERUN_UNVERIFIED {number}/{len(points)} {point_name}",
                flush=True,
            )
        command = [
            sys.executable,
            str(HERE / "retrieval_cache_point.py"),
            "--system",
            system,
            "--nprobe",
            str(nprobe),
            "--arrival-rate",
            str(rate),
            "--topk",
            str(args.topk),
            "--num-requests",
            str(args.num_requests),
            "--workload-mode",
            args.workload_mode,
            "--gpu-id",
            args.gpu_id,
            "--cpu-threads",
            str(args.cpu_threads),
            "--require-clean-resources",
            "--output-dir",
            str(args.output_dir),
        ]
        if args.unique_queries is not None:
            command.extend(["--unique-queries", str(args.unique_queries)])
        print(f"RUN {number}/{len(points)} {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    output = args.output_dir / "campaign.json"
    output.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    references = {
        (row["nprobe"], row["arrival_rate_rps"]): row
        for row in summaries
        if row["system"] == "faiss"
    }
    quality = []
    for row in summaries:
        reference = references[(row["nprobe"], row["arrival_rate_rps"])]
        candidate_ids = np.load(args.output_dir / row["point_name"] / "requests.npz")[
            "returned_doc_ids"
        ]
        reference_ids = np.load(
            args.output_dir / reference["point_name"] / "requests.npz"
        )["returned_doc_ids"]
        overlap = np.mean(
            [
                len(set(candidate.tolist()) & set(target.tolist())) / args.topk
                for candidate, target in zip(candidate_ids, reference_ids)
            ]
        )
        exact = np.mean(np.all(candidate_ids == reference_ids, axis=1))
        quality.append(
            {
                "system": row["system"],
                "nprobe": row["nprobe"],
                "arrival_rate_rps": row["arrival_rate_rps"],
                f"recall_at_{args.topk}_vs_faiss": float(overlap),
                "exact_result_rate": float(exact),
            }
        )
    (args.output_dir / "quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RETRIEVAL_CACHE_CAMPAIGN_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
