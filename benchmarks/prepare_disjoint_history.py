#!/usr/bin/env python3
"""Build leakage-free MnemoRAG history artifacts for formal evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .benchmark_point import (
        ASSET_ROOT,
        DATASET_PATHS,
        load_sample,
        resource_preflight,
    )
    from .fair_workload import array_sha256, file_sha256, select_sample_indices
    from .run_full_e2e_campaign import (
        dataset_row_count,
        question_sha256,
        verify_driver_topology,
    )
    from .runtime_resource_monitor import RuntimeResourceMonitor
except ImportError:
    from benchmark_point import (
        ASSET_ROOT,
        DATASET_PATHS,
        load_sample,
        resource_preflight,
    )
    from fair_workload import array_sha256, file_sha256, select_sample_indices
    from run_full_e2e_campaign import (
        dataset_row_count,
        question_sha256,
        verify_driver_topology,
    )
    from runtime_resource_monitor import RuntimeResourceMonitor


HISTORY_FORMAT = "mnemorag.disjoint-history.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), required=True)
    parser.add_argument("--history-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--faiss-omp-threads", type=int, default=1)
    parser.add_argument("--faiss-search-workers", type=int, default=64)
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-preflight-cpu-busy-percent", type=float, default=5.0)
    parser.add_argument("--max-preflight-cpu-max-busy-percent", type=float, default=5.0)
    parser.add_argument("--max-preflight-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-preflight-gpu-util-percent", type=float, default=5.0)
    parser.add_argument("--max-preflight-load-per-cpu", type=float, default=0.10)
    parser.add_argument(
        "--max-runtime-external-cpu-busy-percent", type=float, default=5.0
    )
    parser.add_argument("--runtime-resource-sample-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-template", type=Path)
    parser.add_argument("--profile-output", type=Path)
    return parser.parse_args()


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def atomic_save_retrieval(
    path: Path,
    retrieval_ids: np.ndarray,
    retrieval_scores: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            retrieval_ids=np.asarray(retrieval_ids, dtype="<i8"),
            retrieval_scores=np.asarray(retrieval_scores, dtype="<f4"),
        )
    temporary.replace(path)


def build_hotness_prior(
    retrieval_ids: np.ndarray,
    *,
    minimum_frequency: int,
) -> dict[str, int]:
    if minimum_frequency < 1:
        raise ValueError("minimum_frequency must be positive")
    ids = np.asarray(retrieval_ids)
    if ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError("history retrieval results must have shape (N, 3)")
    if np.any(ids < 0):
        raise ValueError("history retrieval results contain invalid document IDs")
    counts = Counter(map(int, ids.reshape(-1)))
    return {
        str(doc_id): int(count)
        for doc_id, count in sorted(counts.items())
        if count >= minimum_frequency
    }


def merge_history_into_profile(profile: dict, dataset: str, history: dict) -> dict:
    if profile.get("version") != 1:
        raise ValueError("MnemoRAG profile version must be 1")
    updated = json.loads(json.dumps(profile))
    updated.setdefault("environment", {})["MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR"] = "1"
    dataset_environment = updated.setdefault("dataset_environment", {}).setdefault(
        dataset, {}
    )
    dataset_environment["MNEMORAG_DOC_HOTNESS_PRIOR_PATH"] = history["prior_path"]
    dataset_environment["MNEMORAG_QUERY_BASE_SNAPSHOT_PATH"] = history[
        "query_base_snapshot_path"
    ]
    updated.setdefault("dataset_history", {})[dataset] = history
    return updated


def configure_environment(args: argparse.Namespace) -> list[int]:
    topology = verify_driver_topology(args.gpu_id)
    selected_cpus = list(topology["logical_cpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MNEMORAG_FAISS_OMP_THREADS"] = str(args.faiss_omp_threads)
    os.environ["MNEMORAG_FAISS_SEARCH_WORKERS"] = str(args.faiss_search_workers)
    os.environ["MNEMORAG_FAISS_BATCH_FULL"] = "1"
    os.environ["MNEMORAG_FAISS_PREASSIGNED_GUIDANCE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.faiss_omp_threads)
    args.resource_preflight = resource_preflight(args, selected_cpus)
    return selected_cpus


def retrieve_history(args: argparse.Namespace, rows: list[dict], selected_cpus):
    from mnemorag.config import Config
    from mnemorag.pipeline.utils import QueryPool
    from mnemorag.utils import get_retriever

    config = Config(
        config_dict={
            "split": ["test"],
            "dataset_name": args.dataset,
            "data_dir": str(ASSET_ROOT / "qa/datasets"),
            "index_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
            ),
            "corpus_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
            ),
            "model2path": {"e5": str(ASSET_ROOT / "models/e5-large-v2")},
            "retrieval_method": "e5",
            "retrieval_topk": args.topk,
            "retrieval_batch_size": args.retrieval_batch_size,
            "ivf_nprobe": args.nprobe,
            "enable_similarity_aware": False,
            "enable_preassigned_guidance": False,
            "query_base_capacity": args.history_size,
            "query_base_dtype": "float16",
            "query_base_policy": "lru",
            "query_base_exact_lookup": True,
            "gpu_id": args.gpu_id,
            "gpu_num": 1,
            "cpu_threads": args.cpu_threads,
            "faiss_omp_threads": args.faiss_omp_threads,
            "faiss_search_workers": args.faiss_search_workers,
            "seed": args.seed,
            "disable_save": True,
        }
    )
    retriever = get_retriever(config)
    query_pool = QueryPool(
        device="cuda",
        dtype="float16",
        max_size=args.history_size,
        replacement_policy="lru",
        enable_exact_lookup=True,
    )
    retrieval_ids = np.full((len(rows), args.topk), -1, dtype="<i8")
    retrieval_scores = np.full((len(rows), args.topk), np.nan, dtype="<f4")
    monitor = RuntimeResourceMonitor(
        selected_cpus,
        str(args.gpu_id),
        require_clean=args.require_clean_resources,
        max_external_cpu_busy_percent=args.max_runtime_external_cpu_busy_percent,
        sample_interval_s=args.runtime_resource_sample_seconds,
    )
    started = time.perf_counter()
    monitor.start()
    try:
        for offset in range(0, len(rows), args.retrieval_batch_size):
            batch = rows[offset : offset + args.retrieval_batch_size]
            _docs, scores, ids = retriever.batch_search(
                [row["question"] for row in batch],
                num=args.topk,
                query_pool=query_pool,
                return_score=True,
                return_idx=True,
            )
            count = len(batch)
            retrieval_ids[offset : offset + count] = np.asarray(ids, dtype="<i8")
            retrieval_scores[offset : offset + count] = np.asarray(scores, dtype="<f4")
    finally:
        monitor.checkpoint("history:complete")
        monitor.stop()
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    monitor.assert_clean()
    return (
        retrieval_ids,
        retrieval_scores,
        query_pool,
        time.perf_counter() - started,
        monitor.summary(),
        config,
    )


def main() -> None:
    args = parse_args()
    if args.topk != 3:
        raise ValueError("formal history construction requires topk=3")
    if args.history_size <= 0:
        raise ValueError("history_size must be positive")
    if bool(args.profile_template) != bool(args.profile_output):
        raise ValueError("profile-template and profile-output must be used together")
    selected_cpus = configure_environment(args)
    dataset_path = DATASET_PATHS[args.dataset]
    source_count = dataset_row_count(dataset_path)
    sample_indices = select_sample_indices(
        source_count,
        args.history_size,
        random_seed=args.seed,
    )
    rows, observed_indices, observed_source_count = load_sample(
        dataset_path,
        args.history_size,
        args.seed,
        explicit_indices=sample_indices,
    )
    if (
        observed_source_count != source_count
        or observed_indices != sample_indices.tolist()
    ):
        raise RuntimeError("history sample replay mismatch")

    (
        retrieval_ids,
        retrieval_scores,
        query_pool,
        elapsed_s,
        runtime_resources,
        config,
    ) = retrieve_history(args, rows, selected_cpus)
    prior = build_hotness_prior(retrieval_ids, minimum_frequency=args.minimum_frequency)

    dataset_dir = args.output_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    indices_path = dataset_dir / "history_indices.npy"
    question_hashes_path = dataset_dir / "history_question_hashes.npy"
    retrieval_path = dataset_dir / "history_retrieval.npz"
    prior_path = dataset_dir / "document_hotness_prior.json"
    snapshot_path = dataset_dir / "query_base_snapshot.npz"
    atomic_save_npy(indices_path, sample_indices)
    question_hashes = np.asarray(
        [question_sha256(row["question"]) for row in rows], dtype=np.str_
    )
    atomic_save_npy(question_hashes_path, question_hashes)
    atomic_save_retrieval(retrieval_path, retrieval_ids, retrieval_scores)
    atomic_write_json(prior_path, prior)
    snapshot_metadata = query_pool.save_snapshot(
        snapshot_path,
        metadata={
            "dataset": args.dataset,
            "dataset_sha256": file_sha256(dataset_path),
            "sample_indices_sha256": file_sha256(indices_path),
            "retrieval_ids_sha256": array_sha256(retrieval_ids),
            "retrieval_topk": args.topk,
            "retrieval_nprobe": args.nprobe,
        },
    )
    history = {
        "format": HISTORY_FORMAT,
        "dataset": args.dataset,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": file_sha256(dataset_path),
        "source_count": source_count,
        "history_count": int(sample_indices.size),
        "seed": args.seed,
        "sample_indices_path": str(indices_path.resolve()),
        "sample_indices_sha256": file_sha256(indices_path),
        "question_hashes_path": str(question_hashes_path.resolve()),
        "question_hashes_sha256": file_sha256(question_hashes_path),
        "retrieval_path": str(retrieval_path.resolve()),
        "retrieval_sha256": file_sha256(retrieval_path),
        "retrieval_ids_sha256": array_sha256(retrieval_ids),
        "retrieval_topk": args.topk,
        "retrieval_nprobe": args.nprobe,
        "retrieval_batch_size": args.retrieval_batch_size,
        "prior_path": str(prior_path.resolve()),
        "prior_sha256": file_sha256(prior_path),
        "prior_minimum_frequency": args.minimum_frequency,
        "prior_document_count": len(prior),
        "query_base_snapshot_path": str(snapshot_path.resolve()),
        "query_base_snapshot_sha256": file_sha256(snapshot_path),
        "query_base_entries": query_pool.current_size(),
        "query_base_snapshot_metadata": snapshot_metadata,
        "elapsed_s": elapsed_s,
        "resource_preflight": args.resource_preflight,
        "runtime_resource_monitor": runtime_resources,
        "cpu_topology": verify_driver_topology(args.gpu_id),
        "physical_gpu_id": str(args.gpu_id),
        "index_path": str(Path(config["index_path"]).resolve()),
        "corpus_path": str(Path(config["corpus_path"]).resolve()),
    }
    manifest_path = dataset_dir / "history_manifest.json"
    atomic_write_json(manifest_path, history)
    if args.profile_template:
        profile = json.loads(args.profile_template.read_text(encoding="utf-8"))
        atomic_write_json(
            args.profile_output,
            merge_history_into_profile(profile, args.dataset, history),
        )
    print(json.dumps(history, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
