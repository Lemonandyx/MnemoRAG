#!/usr/bin/env python3
"""Minimal end-to-end MnemoRAG integration test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=os.environ.get("MNEMORAG_ASSET_ROOT"),
        help="Root containing qa/datasets, qa/wiki18_2100w, and models",
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=2)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.asset_root is None:
        raise ValueError("Set MNEMORAG_ASSET_ROOT or pass --asset-root")
    if args.cpu_threads != 64:
        raise ValueError("the paper runtime requires --cpu-threads=64")
    gpu_id = str(args.gpu_id).strip()
    if not gpu_id or "," in gpu_id:
        raise ValueError("--gpu-id must select exactly one GPU")

    available_cpu_ids = sorted(os.sched_getaffinity(0))
    if args.cpu_threads > len(available_cpu_ids):
        raise ValueError(
            f"--cpu-threads={args.cpu_threads} exceeds "
            f"{len(available_cpu_ids)} available CPUs"
        )
    os.sched_setaffinity(0, set(available_cpu_ids[: args.cpu_threads]))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["MNEMORAG_FAISS_OMP_THREADS"] = "1"
    os.environ["MNEMORAG_FAISS_SEARCH_WORKERS"] = str(args.cpu_threads)
    os.environ["MNEMORAG_FAISS_MAX_CODES"] = "0"
    for variable in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PERSISTENT_ASYNC_ENGINE", "1")
    os.environ.setdefault("VLLM_KV_FREE_BLOCK_POLICY", "heat")
    os.environ.setdefault("VLLM_MAX_NUM_SEQS", "8")

    import asyncio
    import json
    import sys
    import time

    from mnemorag.config import Config
    from mnemorag.dataset.dataset import Dataset
    from mnemorag.pipeline import AsyncPipeline
    from mnemorag.prompt import PromptTemplate

    root = args.asset_root.resolve()
    dataset_path = root / "qa/datasets/nq/test.jsonl"
    rows = []
    with dataset_path.open(encoding="utf-8") as source:
        for _ in range(args.num_requests):
            rows.append(json.loads(next(source)))

    config = Config(
        config_dict={
            "split": ["test"],
            "dataset_name": "nq",
            "data_dir": str(root / "qa/datasets"),
            "index_path": str(root / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"),
            "corpus_path": str(root / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"),
            "model2path": {
                "e5": str(root / "models/e5-large-v2"),
                "llama3-8B-instruct": str(root / "models/llama3-8B-instruct"),
            },
            "generator_model": "llama3-8B-instruct",
            "retrieval_method": "e5",
            "framework": "vllm",
            "metrics": ["em", "f1"],
            "disable_save": True,
            "retrieval_topk": args.topk,
            "retrieval_batch_size": args.num_requests,
            "generator_batch_size": max(8, args.num_requests),
            "generator_max_input_len": 2048,
            "generation_params": {"max_tokens": args.max_tokens, "temperature": 0},
            "enable_prefix_caching": True,
            "cpu_threads": args.cpu_threads,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "ivf_nprobe": args.nprobe,
            "enable_similarity_aware": True,
            "enable_hotness_sort": True,
            "gpu_id": gpu_id,
            "gpu_num": 1,
            "irg_max_iter": 3,
            "seed": 0,
        }
    )
    prompt = PromptTemplate(config)

    started = time.perf_counter()
    pipeline = AsyncPipeline(config, prompt_template=prompt)
    output = asyncio.run(pipeline.run_stream(Dataset(data=rows), do_eval=False))
    elapsed = time.perf_counter() - started
    print(f"MNEMORAG_TEST_OK requests={len(output)} elapsed_s={elapsed:.3f}")
    print(f"async_engine={pipeline.generator.use_persistent_async_engine}")
    print(f"hotness_enabled={pipeline.document_pool.enable_hotness}")

    pipeline.generator.model.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
