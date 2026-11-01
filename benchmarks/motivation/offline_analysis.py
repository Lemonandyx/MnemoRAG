#!/usr/bin/env python3
"""Reproduce the data behind MnemoRAG Figures 2(b), 4, and 5.

The script intentionally follows the paper's original helper scripts:

* Figure 4 "Mean" is the mean nearest-other-query cosine similarity.
* Figure 4 Top1/Top5/Top10 are the mean retrieval scores at ranks 1, 5, 10.
* Figure 5 counts accesses to the retrieved Top-3 documents, keeps the 100
  most frequently accessed documents, and normalizes each dataset by its
  maximum access count.
* Figure 2(b) times embedding plus IVF search.  The nprobe curve uses a
  single-query batch; the RBS curve fixes nprobe=128 and times each batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
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
NP_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
RBS_VALUES = NP_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/background_motivation",
    )
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--nn-batch-size", type=int, default=2048)
    parser.add_argument("--timing-single-query-count", type=int, default=32)
    parser.add_argument("--timing-rbs-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314159)
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
            f"Requested {args.cpu_threads} CPU threads but only {len(available)} are available"
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def encode_questions(encoder, questions: list[str], batch_size: int):
    return encoder.encode(questions, batch_size=batch_size, is_query=True)


def retrieve_top10(index, embeddings, batch_size: int):
    import numpy as np

    score_chunks = []
    id_chunks = []
    index.nprobe = 128
    for start in range(0, len(embeddings), batch_size):
        scores, doc_ids = index.search(embeddings[start : start + batch_size], 10)
        score_chunks.append(scores)
        id_chunks.append(doc_ids)
    return np.concatenate(score_chunks), np.concatenate(id_chunks)


def nearest_other_query_similarity(embeddings, batch_size: int):
    import numpy as np
    import torch

    all_vectors = torch.from_numpy(embeddings).to("cuda", dtype=torch.float32)
    # Keep cosine values numerically bounded and comparable with the paper.
    # TF32 can overshoot 1.0 for normalized E5 vectors on Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    maxima = np.empty(len(embeddings), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            stop = min(start + batch_size, len(embeddings))
            similarities = all_vectors[start:stop] @ all_vectors.T
            local_rows = torch.arange(stop - start, device="cuda")
            global_rows = torch.arange(start, stop, device="cuda")
            similarities[local_rows, global_rows] = -float("inf")
            maxima[start:stop] = (
                similarities.max(dim=1).values.clamp_(-1.0, 1.0).cpu().numpy()
            )
            del similarities
    del all_vectors
    torch.cuda.empty_cache()
    return maxima


def figure5_rows(dataset: str, doc_ids) -> list[dict]:
    import numpy as np

    unique_ids, counts = np.unique(doc_ids[:, :3].reshape(-1), return_counts=True)
    order = np.argsort(-counts, kind="stable")[:100]
    ranked_ids = unique_ids[order]
    ranked_counts = counts[order]
    maximum = int(ranked_counts[0]) if len(ranked_counts) else 1
    rows = []
    for rank, (doc_id, count) in enumerate(zip(ranked_ids, ranked_counts), 1):
        rows.append(
            {
                "dataset": dataset,
                "document_rank": rank,
                "document_percentage": (rank - 1) / 99.0
                if len(ranked_ids) > 1
                else 0.0,
                "doc_id": int(doc_id),
                "access_count": int(count),
                "normalized_hotness": float(count) / maximum,
            }
        )
    return rows


def time_figure2b(
    dataset: str,
    questions: list[str],
    encoder,
    index,
    args: argparse.Namespace,
):
    import torch

    rng = random.Random(args.seed + sum(map(ord, dataset)))
    timing_questions = rng.sample(
        questions, min(len(questions), max(512, args.timing_single_query_count + 16))
    )

    # Warm the tokenizer, E5 kernels, and IVF search before measuring.
    warm_embeddings = encode_questions(encoder, timing_questions[:8], 8)
    index.nprobe = 128
    index.search(warm_embeddings, 10)
    sync_cuda(torch)

    rows = []
    for nprobe in NP_VALUES:
        samples_ms = []
        index.nprobe = nprobe
        for question in timing_questions[: args.timing_single_query_count]:
            sync_cuda(torch)
            started = time.perf_counter()
            vector = encode_questions(encoder, [question], 1)
            index.search(vector, 3)
            sync_cuda(torch)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "dataset": dataset,
                "parameter": "nprobe",
                "value": nprobe,
                "fixed_nprobe": "",
                "batch_size": 1,
                "topk": 3,
                "trials": len(samples_ms),
                "mean_latency_ms": statistics.fmean(samples_ms),
                "stdev_latency_ms": statistics.stdev(samples_ms)
                if len(samples_ms) > 1
                else 0.0,
                "median_latency_ms": statistics.median(samples_ms),
            }
        )

    index.nprobe = 128
    cursor = 0
    for rbs in RBS_VALUES:
        samples_ms = []
        for _ in range(args.timing_rbs_repeats):
            if cursor + rbs > len(timing_questions):
                cursor = 0
            batch = timing_questions[cursor : cursor + rbs]
            cursor += rbs
            sync_cuda(torch)
            started = time.perf_counter()
            vectors = encode_questions(encoder, batch, rbs)
            index.search(vectors, 3)
            sync_cuda(torch)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "dataset": dataset,
                "parameter": "rbs",
                "value": rbs,
                "fixed_nprobe": 128,
                "batch_size": rbs,
                "topk": 3,
                "trials": len(samples_ms),
                "mean_latency_ms": statistics.fmean(samples_ms),
                "stdev_latency_ms": statistics.stdev(samples_ms)
                if len(samples_ms) > 1
                else 0.0,
                "median_latency_ms": statistics.median(samples_ms),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    selected_cpus = configure_resources(args)

    import faiss
    import numpy as np
    import torch

    from mnemorag.retriever.encoder import Encoder

    unknown = sorted(set(args.datasets) - set(DATASET_PATHS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    faiss.omp_set_num_threads(args.cpu_threads)
    model_path = ASSET_ROOT / "models/e5-large-v2"
    index_path = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexIVF):
        raise TypeError(f"Expected an IVF index, got {type(index)!r}")
    encoder = Encoder(
        model_name="e5",
        model_path=str(model_path),
        pooling_method="mean",
        max_length=128,
        use_fp16=True,
        instruction=None,
        silent=True,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got {torch.cuda.device_count()}"
        )

    fig4_rows: list[dict] = []
    fig5_all_rows: list[dict] = []
    fig2b_rows: list[dict] = []
    manifest = {
        "paper_figures": ["2b", "4", "5"],
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "physical_gpu_id": str(args.gpu_id),
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "embedding_model": str(model_path),
        "index_path": str(index_path),
        "index_size_bytes": index_path.stat().st_size,
        "index_type": type(index).__name__,
        "nprobe_for_figures_4_5": 128,
        "retrieval_topk_for_figure_4": 10,
        "retrieval_topk_for_figure_5": 3,
        "figure_5_top_documents": 100,
        "dataset_files": {},
    }

    for dataset in args.datasets:
        path = DATASET_PATHS[dataset]
        if not path.exists():
            raise FileNotFoundError(path)
        questions = load_questions(path)
        manifest["dataset_files"][dataset] = {
            "path": str(path),
            "queries": len(questions),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        print(f"DATASET_START name={dataset} queries={len(questions)}", flush=True)

        embedding_path = cache_dir / f"{dataset}_query_embeddings.npy"
        if embedding_path.exists():
            embeddings = np.load(embedding_path)
            if embeddings.shape != (len(questions), 1024):
                raise ValueError(
                    f"Unexpected cached embedding shape: {embeddings.shape}"
                )
        else:
            embeddings = encode_questions(encoder, questions, args.encoding_batch_size)
            np.save(embedding_path, embeddings)

        score_path = cache_dir / f"{dataset}_nprobe128_top10_scores.npy"
        id_path = cache_dir / f"{dataset}_nprobe128_top10_doc_ids.npy"
        if score_path.exists() and id_path.exists():
            scores = np.load(score_path)
            doc_ids = np.load(id_path)
        else:
            scores, doc_ids = retrieve_top10(
                index, embeddings, args.retrieval_batch_size
            )
            np.save(score_path, scores)
            np.save(id_path, doc_ids)
        if scores.shape != (len(questions), 10) or doc_ids.shape != scores.shape:
            raise ValueError(
                f"Unexpected retrieval shapes for {dataset}: {scores.shape}, {doc_ids.shape}"
            )

        nearest_path = cache_dir / f"{dataset}_nearest_other_similarity_fp32.npy"
        if nearest_path.exists():
            nearest = np.load(nearest_path)
        else:
            nearest = nearest_other_query_similarity(embeddings, args.nn_batch_size)
            np.save(nearest_path, nearest)

        fig4_rows.append(
            {
                "dataset": dataset,
                "num_queries": len(questions),
                "top1_retrieval_similarity": float(scores[:, 0].mean()),
                "top5_retrieval_similarity": float(scores[:, 4].mean()),
                "top10_retrieval_similarity": float(scores[:, 9].mean()),
                "mean_inter_request_similarity": float(nearest.mean()),
                "min_nearest_query_similarity": float(nearest.min()),
                "max_nearest_query_similarity": float(nearest.max()),
            }
        )
        fig5_all_rows.extend(figure5_rows(dataset, doc_ids))
        fig2b_rows.extend(time_figure2b(dataset, questions, encoder, index, args))

        write_csv(
            args.output_dir / "figure4_semantic_similarity.csv",
            fig4_rows,
            list(fig4_rows[0]),
        )
        write_csv(
            args.output_dir / "figure5_document_hotness.csv",
            fig5_all_rows,
            list(fig5_all_rows[0]),
        )
        write_csv(
            args.output_dir / "figure2b_retrieval_latency.csv",
            fig2b_rows,
            list(fig2b_rows[0]),
        )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"DATASET_DONE name={dataset}", flush=True)

    print("OFFLINE_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
