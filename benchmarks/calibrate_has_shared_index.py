#!/usr/bin/env python3
"""Calibrate HaS homology threshold on NQ train with the shared IVF index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--backend-nprobe", type=int, default=512)
    parser.add_argument("--fuzzy-nprobe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results/evaluation/retrieval_cache_scaling/has_shared_index_calibration.json",
    )
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[: args.cpu_threads]))
    faiss.omp_set_num_threads(args.cpu_threads)
    has_root = Path(
        os.environ.get("HAS_ROOT", PROJECT_ROOT / "third_party/HaS")
    ).resolve()
    sys.path.insert(0, str(has_root / "src"))
    from SpeculativeRetrievalService.HaS import CacheChannel
    from utils.index import HNSWLibIndexerV2

    all_queries = np.load(QUERY_EMBEDDINGS, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(len(all_queries), args.num_queries, replace=False))
    queries = np.asarray(all_queries[selected], dtype=np.float32)
    faiss.normalize_L2(queries)
    docs = np.load(DOC_VECTORS, mmap_mode="r")
    index = faiss.read_index(str(INDEX_PATH))
    index.nprobe = args.backend_nprobe
    _, reference_ids = index.search(queries, args.topk)
    index.nprobe = args.fuzzy_nprobe
    _, fuzzy_ids = index.search(queries, args.topk * 5)
    fuzzy_ranked = np.empty((args.num_queries, args.topk), dtype=np.int64)
    fuzzy_scores = np.empty((args.num_queries, args.topk), dtype=np.float32)
    for row, (query, ids) in enumerate(zip(queries, fuzzy_ids)):
        vectors = np.asarray(docs[ids], dtype=np.float32)
        scores = vectors @ query
        order = np.argsort(-scores)[: args.topk]
        fuzzy_ranked[row] = ids[order]
        fuzzy_scores[row] = scores[order]

    results = []
    for threshold in (0.2, 0.4, 0.6, 0.8, 1.0):
        cache = CacheChannel()
        cache.index = HNSWLibIndexerV2(vector_sz=1024)
        cache.reset(100_000)
        outputs = np.empty_like(reference_ids)
        hits = 0
        for row, query in enumerate(queries):
            cached_docs = cache.search(query, top_n=args.topk)
            merged = {
                int(doc_id): float(score)
                for doc_id, score in zip(fuzzy_ranked[row], fuzzy_scores[row])
            }
            for doc in cached_docs:
                merged[doc["id"]] = max(merged.get(doc["id"], -np.inf), doc["score"])
            candidate = np.asarray(
                [
                    doc_id
                    for doc_id, _ in sorted(
                        merged.items(), key=lambda item: item[1], reverse=True
                    )[: args.topk]
                ],
                dtype=np.int64,
            )
            previous, _ = cache.is_similar_query_exist(candidate.tolist(), threshold)
            if previous is not None:
                outputs[row] = candidate
                hits += 1
            else:
                outputs[row] = reference_ids[row]
                cache.add(
                    {"id": row},
                    reference_ids[row],
                    np.asarray(docs[reference_ids[row]], dtype=np.float32),
                )
        recall = np.mean(
            [
                len(set(candidate.tolist()) & set(reference.tolist())) / args.topk
                for candidate, reference in zip(outputs, reference_ids)
            ]
        )
        exact = np.mean(np.all(outputs == reference_ids, axis=1))
        result = {
            "threshold": threshold,
            "hit_rate": hits / args.num_queries,
            f"recall_at_{args.topk}": float(recall),
            "exact_result_rate": float(exact),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    payload = {
        "dataset": "NQ train held-out sample",
        "num_queries": args.num_queries,
        "topk": args.topk,
        "backend_index": str(INDEX_PATH),
        "backend_nprobe": args.backend_nprobe,
        "fuzzy_nprobe": args.fuzzy_nprobe,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
