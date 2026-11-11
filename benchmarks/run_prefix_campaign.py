#!/usr/bin/env python3
"""Measure hotness ordering versus native vLLM LRU prefix reuse (Figure 14 subset)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .experiment_validation import verified_complete, verified_evaluation_contract
except ImportError:
    from experiment_validation import verified_complete, verified_evaluation_contract


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation/prefix_hit",
    )
    args = parser.parse_args()
    points = [
        (label, components, topk)
        for label, components in (
            ("mnemorag_hotness", "sar_hotness"),
            ("vllm_lru", "sar"),
        )
        for topk in (1, 3, 5, 10)
    ]
    summaries = []
    for number, (label, components, topk) in enumerate(points, 1):
        point_dir = args.output_dir / label
        point_name = f"nq_onepass_{components}_k{topk}_np128_rps24_n{args.num_requests}"
        summary_path = point_dir / point_name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if verified_complete(
                summary,
                args.cpu_threads,
                ("prefix_request_hit_rate_mean", "prefix_request_hit_rate_max"),
                expected_point={
                    "dataset": "nq",
                    "workflow": "onepass",
                    "components": components,
                    "order_mode": "hotness_hint",
                    "num_requests": args.num_requests,
                    "arrival_rate_rps": 24.0,
                    "topk": topk,
                    "nprobe": 128,
                    "seed": 20260923,
                },
            ) and verified_evaluation_contract(
                summary, topk, ordering_mode="hotness_hint"
            ):
                print(
                    f"SKIP_COMPLETE {number}/{len(points)} {label} topk={topk}",
                    flush=True,
                )
                summaries.append(summary)
                continue
            print(
                f"RERUN_UNVERIFIED {number}/{len(points)} {label} topk={topk}",
                flush=True,
            )
        command = [
            sys.executable,
            str(HERE / "benchmark_point.py"),
            "--dataset",
            "nq",
            "--workflow",
            "onepass",
            "--components",
            components,
            "--topk",
            str(topk),
            "--nprobe",
            "128",
            "--arrival-rate",
            "24",
            "--num-requests",
            str(args.num_requests),
            "--record-prefix-distribution",
            "--gpu-id",
            args.gpu_id,
            "--cpu-threads",
            str(args.cpu_threads),
            "--require-clean-resources",
            "--output-dir",
            str(point_dir),
        ]
        print(
            f"RUN_PREFIX_POINT {number}/{len(points)}: {' '.join(command)}", flush=True
        )
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["prefix_policy_label"] = label
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "figure14_campaign.json"
    output.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PREFIX_CAMPAIGN_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
