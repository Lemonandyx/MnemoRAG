#!/usr/bin/env python3
"""Calibrate QVCache tolerance on NQ train with the common IVF backend."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
BASELINE_ARTIFACTS = Path(
    os.environ.get(
        "MNEMORAG_BASELINE_ARTIFACTS",
        PROJECT_ROOT / "results/retrieval_baseline_artifacts",
    )
).resolve()
INDEX_PATH = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
QUERY_EMBEDDINGS = Path(
    os.environ.get(
        "MNEMORAG_NQ_EMBEDDINGS",
        PROJECT_ROOT / "results/background_motivation/cache/nq_query_embeddings.npy",
    )
).resolve()
DOC_VECTORS = Path(
    os.environ.get(
        "MNEMORAG_VECTOR_PATH",
        PROJECT_ROOT / "assets/wiki18_2100w/vectors.f16.npy",
    )
).resolve()
QV_SAMPLE = BASELINE_ARTIFACTS / "wiki18_e5_qvcache_pca_sample.f32.bin"
QV_PCA_PREFIX = BASELINE_ARTIFACTS / "wiki18_e5_qvcache"


class Backend:
    def __init__(self, index, vectors) -> None:
        self.index = index
        self.vectors = vectors

    def search(self, query: np.ndarray, k: int):
        scores, ids = self.index.search(np.asarray(query, dtype=np.float32)[None, :], k)
        return ids[0].astype(np.uint32), (1.0 - scores[0]).astype(np.float32)

    def fetch_vectors_by_ids(self, ids: list[int]):
        rows = np.asarray(
            self.vectors[np.asarray(ids, dtype=np.int64)], dtype=np.float32
        )
        return [row for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--arrival-rate", type=float, default=28.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results/evaluation/retrieval_cache_scaling/qvcache_shared_index_calibration.json",
    )
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[: args.cpu_threads]))
    faiss.omp_set_num_threads(args.cpu_threads)
    qvcache_root = Path(
        os.environ.get("QVCACHE_ROOT", PROJECT_ROOT / "third_party/QVCache")
    ).resolve()
    sys.path.insert(0, str(qvcache_root / "python"))
    import qvcache as qvc

    all_queries = np.load(QUERY_EMBEDDINGS, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(len(all_queries), args.num_queries, replace=False))
    queries = np.asarray(all_queries[selected], dtype=np.float32)
    faiss.normalize_L2(queries)
    vectors = np.load(DOC_VECTORS, mmap_mode="r")
    index = faiss.read_index(str(INDEX_PATH))
    index.nprobe = args.nprobe
    _, reference_ids = index.search(queries, args.topk)
    results = []
    for deviation in (0.05, 0.10, 0.15, 0.25):
        backend = Backend(index, vectors)
        cache = qvc.QVCache(
            data_path=str(QV_SAMPLE),
            pca_prefix=str(QV_PCA_PREFIX),
            R=64,
            memory_L=32,
            B=8,
            M=8,
            alpha=1.2,
            build_threads=8,
            search_threads=24,
            use_reconstructed_vectors=True,
            p=0.90,
            deviation_factor=deviation,
            memory_index_max_points=100_000,
            beamwidth=2,
            use_regional_theta=True,
            pca_dim=16,
            buckets_per_dim=8,
            max_regions=1_000_000,
            n_async_insert_threads=4,
            lazy_theta_updates=True,
            number_of_mini_indexes=4,
            search_mini_indexes_in_parallel=False,
            max_search_threads=32,
            metric=qvc.Metric.COSINE,
            backend=backend,
        )
        cache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
        cache.enable_adaptive_strategy(True)
        cache.set_hit_ratio_window_size(100)
        cache.set_hit_ratio_threshold(0.90)
        outputs = np.empty_like(reference_ids)
        hits = 0
        for row, query in enumerate(queries):
            hit, ids, _ = cache.search(query, args.topk)
            outputs[row] = np.asarray(ids, dtype=np.int64) - 1
            hits += int(hit)
            time.sleep(1.0 / args.arrival_rate)
        recall = np.mean(
            [
                len(set(candidate.tolist()) & set(reference.tolist())) / args.topk
                for candidate, reference in zip(outputs, reference_ids)
            ]
        )
        exact = np.mean(np.all(outputs == reference_ids, axis=1))
        result = {
            "deviation_factor": deviation,
            "hit_rate": hits / args.num_queries,
            f"recall_at_{args.topk}": float(recall),
            "exact_result_rate": float(exact),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del cache
        gc.collect()
    payload = {
        "dataset": "NQ train held-out sample",
        "num_queries": args.num_queries,
        "topk": args.topk,
        "backend_index": str(INDEX_PATH),
        "nprobe": args.nprobe,
        "arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
