#!/usr/bin/env python3
"""Recompute document hotness from every query's complete FAISS top-100.

For each dataset, every query is searched exactly once with ``topk=100`` and
``nprobe=128``. All 100 returned document IDs are counted, then the 100 most
frequently returned documents are written in descending access-count order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
DATASET_PATHS = {
    "nq": ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    "hotpotqa": ASSET_ROOT / "qa/datasets/hotpotqa/train.jsonl",
    "squad": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
    "2wikimultihopqa": ASSET_ROOT / "qa/datasets/2wikimultihopqa/train.jsonl",
}
EXPECTED_QUERY_COUNTS = {
    "nq": 79_168,
    "hotpotqa": 90_447,
    "squad": 10_570,
    "2wikimultihopqa": 15_000,
}
DEFAULT_MODEL_PATH = ASSET_ROOT / "models/e5-large-v2"
DEFAULT_INDEX_PATH = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
DEFAULT_EMBEDDING_CACHE = PROJECT_ROOT / "results/background_motivation/cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/background_motivation/document_hotness"
NPROBE = 128
RETRIEVAL_TOPK = 100
TOP_DOCUMENTS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count every document returned by per-query FAISS top-100 retrieval."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedding-cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--force-retrieval", action="store_true")
    return parser.parse_args()


def configure_resources(args: argparse.Namespace) -> list[int]:
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    gpu_id = str(args.gpu_id).strip()
    if not gpu_id or "," in gpu_id:
        raise ValueError("--gpu-id must select exactly one GPU")
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads > len(available):
        raise ValueError(
            f"Requested {args.cpu_threads} CPU threads but only "
            f"{len(available)} are available"
        )
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    for name in (
        "MNEMORAG_FAISS_OMP_THREADS",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return selected


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question in {path}:{line_number}")
            questions.append(question.strip())
    return questions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save_npy(path: Path, value) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    temporary.replace(path)


def load_or_encode_embeddings(
    dataset: str,
    questions: list[str],
    args: argparse.Namespace,
):
    import numpy as np

    path = args.embedding_cache_dir / f"{dataset}_query_embeddings.npy"
    if path.exists():
        embeddings = np.load(path, mmap_mode="r")
        if embeddings.shape != (len(questions), 1024):
            raise ValueError(
                f"Unexpected cached embedding shape for {dataset}: {embeddings.shape}"
            )
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Expected float32 cached embeddings for {dataset}, got {embeddings.dtype}"
            )
        print(f"EMBEDDINGS_REUSED dataset={dataset} path={path}", flush=True)
        return embeddings, path

    import torch

    from mnemorag.retriever.encoder import Encoder

    args.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    encoder = Encoder(
        model_name="e5",
        model_path=str(args.model_path),
        pooling_method="mean",
        max_length=128,
        use_fp16=True,
        instruction=None,
        silent=True,
    )
    embeddings = encoder.encode(
        questions,
        batch_size=args.encoding_batch_size,
        is_query=True,
    )
    if embeddings.shape != (len(questions), 1024):
        raise ValueError(f"Unexpected encoded embedding shape: {embeddings.shape}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, embeddings)
    temporary.replace(path)
    print(f"EMBEDDINGS_CREATED dataset={dataset} path={path}", flush=True)
    return np.load(path, mmap_mode="r"), path


def count_top100_accesses(
    index,
    embeddings,
    batch_size: int,
    dataset: str,
    partial_count_path: Path,
    progress_path: Path,
):
    import numpy as np

    start_offset = 0
    previous_elapsed = 0.0
    if partial_count_path.exists() and progress_path.exists():
        counts = np.load(partial_count_path)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_offset = int(progress["processed_queries"])
        total_valid = int(progress["total_valid_records"])
        previous_elapsed = float(progress.get("retrieval_seconds", 0.0))
        if counts.shape != (index.ntotal,) or counts.dtype != np.uint32:
            raise ValueError(
                f"Unexpected partial count cache for {dataset}: "
                f"shape={counts.shape}, dtype={counts.dtype}"
            )
        if not 0 <= start_offset <= len(embeddings):
            raise ValueError(f"Invalid resume offset for {dataset}: {start_offset}")
        if total_valid != start_offset * RETRIEVAL_TOPK:
            raise ValueError(
                f"Invalid resume record count for {dataset}: {total_valid}"
            )
        if int(counts.sum(dtype=np.uint64)) != total_valid:
            raise ValueError(f"Partial access-count sum mismatch for {dataset}")
        print(
            f"RETRIEVAL_RESUME dataset={dataset} queries={start_offset}/{len(embeddings)} "
            f"records={total_valid}",
            flush=True,
        )
    elif partial_count_path.exists() or progress_path.exists():
        raise RuntimeError(
            f"Incomplete resume checkpoint for {dataset}: expected both "
            f"{partial_count_path} and {progress_path}"
        )
    else:
        counts = np.zeros(index.ntotal, dtype=np.uint32)
        total_valid = 0

    started = time.perf_counter()
    for start in range(start_offset, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        query_vectors = np.ascontiguousarray(embeddings[start:stop], dtype=np.float32)
        _, doc_ids = index.search(query_vectors, RETRIEVAL_TOPK)
        valid_ids = doc_ids[doc_ids >= 0]
        if valid_ids.size:
            unique_ids, batch_counts = np.unique(valid_ids, return_counts=True)
            counts[unique_ids] += batch_counts.astype(np.uint32, copy=False)
        total_valid += int(valid_ids.size)
        if stop == len(embeddings) or stop % 10_000 < batch_size:
            elapsed = previous_elapsed + time.perf_counter() - started
            atomic_save_npy(partial_count_path, counts)
            atomic_write_json(
                progress_path,
                {
                    "dataset": dataset,
                    "processed_queries": stop,
                    "total_queries": len(embeddings),
                    "retrieval_topk": RETRIEVAL_TOPK,
                    "nprobe": NPROBE,
                    "total_valid_records": total_valid,
                    "retrieval_seconds": elapsed,
                },
            )
            print(
                f"RETRIEVAL_PROGRESS dataset={dataset} queries={stop}/{len(embeddings)} "
                f"records={total_valid} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    expected = len(embeddings) * RETRIEVAL_TOPK
    if total_valid != expected:
        raise RuntimeError(
            f"Expected {expected} valid retrieval IDs for {dataset}, got {total_valid}"
        )
    if int(counts.sum(dtype=np.uint64)) != expected:
        raise RuntimeError(f"Access-count sum mismatch for {dataset}")
    return counts, previous_elapsed + time.perf_counter() - started


def rank_top_documents(dataset: str, counts) -> tuple[list[dict], dict]:
    import numpy as np

    if counts.size < TOP_DOCUMENTS:
        raise ValueError("Index contains fewer than 100 documents")
    candidates = np.argpartition(counts, -TOP_DOCUMENTS)[-TOP_DOCUMENTS:]
    candidate_counts = counts[candidates].astype(np.uint64)
    order = np.lexsort((candidates, -candidate_counts.astype(np.int64)))
    ranked_ids = candidates[order]
    ranked_counts = candidate_counts[order]
    maximum = int(ranked_counts[0])
    total_accesses = int(counts.sum(dtype=np.uint64))
    top100_total = int(ranked_counts.sum(dtype=np.uint64))
    cumulative = 0
    rows: list[dict] = []
    for rank, (doc_id, count_value) in enumerate(zip(ranked_ids, ranked_counts), 1):
        count = int(count_value)
        cumulative += count
        rows.append(
            {
                "dataset": dataset,
                "document_rank": rank,
                "document_percentage": (rank - 1) / (TOP_DOCUMENTS - 1),
                "doc_id": int(doc_id),
                "access_count": count,
                "normalized_hotness": count / maximum if maximum else 0.0,
                "access_share_all_top100_results": count / total_accesses,
                "cumulative_access_share_all_top100_results": cumulative
                / total_accesses,
                "access_share_within_hottest_100": count / top100_total,
                "cumulative_access_share_within_hottest_100": cumulative / top100_total,
            }
        )
    summary = {
        "total_access_records": total_accesses,
        "unique_retrieved_documents": int(np.count_nonzero(counts)),
        "hottest_document_id": int(ranked_ids[0]),
        "hottest_document_access_count": maximum,
        "rank_100_access_count": int(ranked_counts[-1]),
        "hottest_100_access_records": top100_total,
        "hottest_100_coverage_pct": top100_total / total_accesses * 100.0,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.datasets) - set(DATASET_PATHS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    selected_cpus = configure_resources(args)

    import faiss
    import numpy as np

    for required_path in (args.index_path, args.model_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count_cache_dir = args.output_dir / "cache"
    count_cache_dir.mkdir(parents=True, exist_ok=True)

    faiss.omp_set_num_threads(args.cpu_threads)
    print(f"INDEX_LOAD_START path={args.index_path}", flush=True)
    index = faiss.read_index(str(args.index_path))
    if not isinstance(index, faiss.IndexIVF):
        raise TypeError(f"Expected an IVF index, got {type(index)!r}")
    if index.nlist != 4096:
        raise ValueError(f"Expected nlist=4096, got {index.nlist}")
    index.nprobe = NPROBE
    print(
        f"INDEX_LOAD_DONE type={type(index).__name__} ntotal={index.ntotal} "
        f"nlist={index.nlist} nprobe={index.nprobe}",
        flush=True,
    )

    output_csv = args.output_dir / "figure5_document_hotness_retrieval_top100.csv"
    summary_csv = (
        args.output_dir / "figure5_document_hotness_retrieval_top100_summary.csv"
    )
    manifest_path = args.output_dir / "manifest_retrieval_top100.json"
    previous_manifest = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    manifest = {
        "definition": (
            "Each query is searched once with FAISS topk=100 and nprobe=128. "
            "Every returned document ID is counted. The 100 documents with the "
            "largest aggregate access counts are reported for each dataset."
        ),
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "physical_gpu_id": str(args.gpu_id),
        "embedding_model": str(args.model_path),
        "index_path": str(args.index_path),
        "index_size_bytes": args.index_path.stat().st_size,
        "index_type": type(index).__name__,
        "index_ntotal": int(index.ntotal),
        "index_nlist": int(index.nlist),
        "nprobe": NPROBE,
        "retrieval_topk_per_query": RETRIEVAL_TOPK,
        "reported_hottest_documents_per_dataset": TOP_DOCUMENTS,
        "datasets": {},
    }

    for dataset in args.datasets:
        dataset_path = DATASET_PATHS[dataset]
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        questions = load_questions(dataset_path)
        expected_queries = EXPECTED_QUERY_COUNTS[dataset]
        if len(questions) != expected_queries:
            raise ValueError(
                f"Expected {expected_queries} queries for {dataset}, got {len(questions)}"
            )
        embeddings, embedding_path = load_or_encode_embeddings(dataset, questions, args)
        print(f"DATASET_START name={dataset} queries={len(questions)}", flush=True)

        count_path = count_cache_dir / f"{dataset}_nprobe128_top100_access_counts.npy"
        partial_count_path = count_cache_dir / (
            f"{dataset}_nprobe128_top100_access_counts.partial.npy"
        )
        progress_path = count_cache_dir / f"{dataset}_nprobe128_top100_progress.json"
        if count_path.exists() and not args.force_retrieval:
            counts = np.load(count_path, mmap_mode="r")
            if counts.shape != (index.ntotal,) or counts.dtype != np.uint32:
                raise ValueError(
                    f"Unexpected count cache for {dataset}: "
                    f"shape={counts.shape}, dtype={counts.dtype}"
                )
            retrieval_seconds = (
                previous_manifest.get("datasets", {})
                .get(dataset, {})
                .get("retrieval_seconds")
            )
            print(f"COUNTS_REUSED dataset={dataset} path={count_path}", flush=True)
        else:
            if args.force_retrieval:
                partial_count_path.unlink(missing_ok=True)
                progress_path.unlink(missing_ok=True)
            counts, retrieval_seconds = count_top100_accesses(
                index,
                embeddings,
                args.retrieval_batch_size,
                dataset,
                partial_count_path,
                progress_path,
            )
            atomic_save_npy(count_path, counts)
            partial_count_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            counts = np.load(count_path, mmap_mode="r")

        rows, summary = rank_top_documents(dataset, counts)
        all_rows.extend(rows)
        summary_row = {
            "dataset": dataset,
            "num_queries": len(questions),
            "retrieval_topk_per_query": RETRIEVAL_TOPK,
            "nprobe": NPROBE,
            **summary,
            "retrieval_seconds": ""
            if retrieval_seconds is None
            else f"{retrieval_seconds:.6f}",
        }
        summary_rows.append(summary_row)
        manifest["datasets"][dataset] = {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "sha256": sha256(dataset_path),
            "queries": len(questions),
            "expected_access_records": len(questions) * RETRIEVAL_TOPK,
            "embedding_cache": str(embedding_path),
            "access_count_cache": str(count_path),
            **summary,
            "retrieval_seconds": retrieval_seconds,
        }

        atomic_write_csv(output_csv, all_rows, list(all_rows[0]))
        atomic_write_csv(summary_csv, summary_rows, list(summary_rows[0]))
        atomic_write_json(manifest_path, manifest)
        print(
            f"DATASET_DONE name={dataset} records={summary['total_access_records']} "
            f"unique_docs={summary['unique_retrieved_documents']} "
            f"top100_coverage_pct={summary['hottest_100_coverage_pct']:.6f}",
            flush=True,
        )

    expected_total = sum(EXPECTED_QUERY_COUNTS[name] for name in args.datasets) * 100
    actual_total = sum(row["total_access_records"] for row in summary_rows)
    if actual_total != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} access records across datasets, got {actual_total}"
        )
    print(
        f"TOP100_HOTNESS_OK datasets={len(args.datasets)} "
        f"queries={sum(EXPECTED_QUERY_COUNTS[name] for name in args.datasets)} "
        f"access_records={actual_total} output={output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
