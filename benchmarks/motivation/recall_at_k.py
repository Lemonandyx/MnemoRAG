#!/usr/bin/env python3
"""Measure Recall@100 while varying IVF nprobe.

The exact reference is the same IVF4096-Flat index searched with nprobe=4096,
which scans every inverted list.  Query samples are reused from Figure 2(a).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

NP_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DATASETS = ("nq", "hotpotqa", "squad", "2wikimultihopqa")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/background_motivation",
    )
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def search_in_batches(index, embeddings, k: int, batch_size: int):
    import numpy as np

    ids = []
    elapsed = 0.0
    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start : start + batch_size]
        began = time.perf_counter()
        _, batch_ids = index.search(batch, k)
        elapsed += time.perf_counter() - began
        ids.append(batch_ids)
    return np.concatenate(ids), elapsed


def recall_per_query(candidate_ids, exact_ids):
    values = []
    for candidate, exact in zip(candidate_ids, exact_ids):
        values.append(len(set(map(int, candidate)) & set(map(int, exact))) / len(exact))
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.k != 100:
        raise ValueError("This reproduction is fixed to k=100")
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads > len(available):
        raise ValueError("Insufficient available CPU cores")
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    for name in (
        "MNEMORAG_FAISS_OMP_THREADS",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)

    import faiss
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    index_path = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
    faiss.omp_set_num_threads(args.cpu_threads)
    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexIVF) or index.nlist != 4096:
        raise TypeError(
            f"Expected IVF4096 index, got {type(index)!r}, nlist={index.nlist}"
        )

    result_rows: list[dict] = []
    manifest = {
        "metric": "Recall@100",
        "k": args.k,
        "nprobe_values": list(NP_VALUES),
        "ground_truth_nprobe": index.nlist,
        "ground_truth_definition": "IVF4096-Flat searched over all 4096 lists",
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected,
        "physical_gpu_id": str(args.gpu_id),
        "sample_source": "Figure 2(a) manifests",
        "datasets": {},
    }

    for dataset in DATASETS:
        figure2a_manifest = json.loads(
            (args.output_dir / f"figure2a_{dataset}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        sample_indices = np.asarray(figure2a_manifest["sample_indices"], dtype=np.int64)
        embeddings = np.load(
            cache_dir / f"{dataset}_query_embeddings.npy", mmap_mode="r"
        )
        sample_embeddings = np.ascontiguousarray(
            embeddings[sample_indices], dtype=np.float32
        )

        index.nprobe = index.nlist
        exact_ids, exact_seconds = search_in_batches(
            index, sample_embeddings, args.k, args.batch_size
        )
        if (exact_ids < 0).any():
            raise RuntimeError(f"Exact search returned invalid IDs for {dataset}")

        manifest["datasets"][dataset] = {
            "sample_size": len(sample_indices),
            "sample_indices": sample_indices.tolist(),
            "exact_search_seconds": exact_seconds,
        }
        print(
            f"EXACT_DONE dataset={dataset} queries={len(sample_indices)} "
            f"seconds={exact_seconds:.3f}",
            flush=True,
        )

        for nprobe in NP_VALUES:
            index.nprobe = nprobe
            candidate_ids, search_seconds = search_in_batches(
                index, sample_embeddings, args.k, args.batch_size
            )
            recalls = recall_per_query(candidate_ids, exact_ids)
            result_rows.append(
                {
                    "dataset": dataset,
                    "num_queries": len(sample_indices),
                    "k": args.k,
                    "nprobe": nprobe,
                    "ground_truth_nprobe": index.nlist,
                    "mean_recall_at_100": statistics.fmean(recalls),
                    "stdev_recall_at_100": statistics.stdev(recalls),
                    "min_recall_at_100": min(recalls),
                    "max_recall_at_100": max(recalls),
                    "batch_size": args.batch_size,
                    "search_seconds": search_seconds,
                    "mean_search_ms_per_query": 1000
                    * search_seconds
                    / len(sample_indices),
                }
            )
            print(
                f"RECALL_DONE dataset={dataset} nprobe={nprobe} "
                f"recall={result_rows[-1]['mean_recall_at_100']:.6f}",
                flush=True,
            )
        write_csv(args.output_dir / "figure2c_recall_at_100.csv", result_rows)
        (args.output_dir / "figure2c_recall_at_100_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("RECALL_AT_100_OK", flush=True)


if __name__ == "__main__":
    main()
