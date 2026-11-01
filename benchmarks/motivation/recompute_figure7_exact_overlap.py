#!/usr/bin/env python3
"""Recompute MnemoRAG Figure 7 with exact document/probe semantics.

Panel (a): on the complete 79,168-query NQ train split, pair every query
with its globally most similar *other* query and measure direct Top-3/5/10
document-ID set overlap across similarity thresholds.

Panel (b): on four complete evaluation splits, pair every query with its
globally most similar other query, measure overlap between their ordered
Top-128 IVF probe sets, and measure the 1-based probe positions of each
query's retrieved Top-3/5/10 documents.
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
INDEX_PATH = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
MODEL_PATH = ASSET_ROOT / "models/e5-large-v2"
NQ_TRAIN = ASSET_ROOT / "qa/datasets/nq/train.jsonl"

EVAL_DATASETS = {
    "nq": {
        "display": "NQ",
        "path": ASSET_ROOT / "qa/datasets/nq/test.jsonl",
        "expected_queries": 3610,
        "source_split": "test",
    },
    "hotpotqa": {
        "display": "HotpotQA",
        "path": ASSET_ROOT / "qa/datasets/hotpotqa/dev.jsonl",
        "expected_queries": 7405,
        "source_split": "dev (complete public evaluation split; test alias)",
    },
    "squad": {
        "display": "SQuAD",
        "path": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
        "expected_queries": 10570,
        "source_split": "dev (complete public evaluation split; test alias)",
    },
    "2wikimultihopqa": {
        "display": "2WikiMultiHopQA",
        "path": ASSET_ROOT / "qa/datasets/2wikimultihopqa/test.jsonl",
        "expected_queries": 12576,
        "source_split": "dev (complete public evaluation split; test alias)",
    },
}

NPROBE = 128
MAX_TOPK = 10
TOPK_VALUES = (3, 5, 10)
THRESHOLDS = tuple(round(0.90 + 0.01 * index, 2) for index in range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/background_motivation/overlap_rank",
    )
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--nn-batch-size", type=int, default=1024)
    return parser.parse_args()


def configure_resources(args: argparse.Namespace) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads > len(available):
        raise ValueError(f"Only {len(available)} CPUs are available")
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MNEMORAG_FAISS_OMP_THREADS",
    ):
        os.environ[key] = str(args.cpu_threads)
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> list[str]:
    questions = []
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def encode_or_load(encoder, questions: list[str], path: Path, batch_size: int):
    import numpy as np

    expected = (len(questions), 1024)
    if path.exists():
        array = np.load(path, mmap_mode="r")
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(
                f"Invalid embedding cache {path}: {array.shape}, {array.dtype}"
            )
        print(f"EMBEDDINGS_REUSED path={path} shape={array.shape}", flush=True)
        return array
    began = time.perf_counter()
    array = encoder.encode(questions, batch_size=batch_size, is_query=True)
    array = np.asarray(array, dtype=np.float32)
    if array.shape != expected:
        raise ValueError(f"Unexpected encoded shape {array.shape}, expected {expected}")
    np.save(path, array)
    print(
        f"EMBEDDINGS_DONE path={path} seconds={time.perf_counter() - began:.1f}",
        flush=True,
    )
    return np.load(path, mmap_mode="r")


def retrieve_or_load(index, embeddings, path: Path, state_path: Path, batch_size: int):
    import numpy as np

    expected = (len(embeddings), MAX_TOPK)
    completed = 0
    if path.exists():
        result = np.load(path, mmap_mode="r+")
        if result.shape != expected or result.dtype != np.int64:
            raise ValueError(
                f"Invalid retrieval cache {path}: {result.shape}, {result.dtype}"
            )
        completed = (
            int(json.loads(state_path.read_text()).get("completed_rows", 0))
            if state_path.exists()
            else len(result)
        )
    else:
        result = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.int64, shape=expected
        )
        result.fill(-1)
        result.flush()
    index.nprobe = NPROBE
    for start in range(completed, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        _, ids = index.search(np.ascontiguousarray(embeddings[start:stop]), MAX_TOPK)
        result[start:stop] = ids
        result.flush()
        atomic_json(
            state_path,
            {
                "completed_rows": stop,
                "query_count": len(embeddings),
                "nprobe": NPROBE,
                "topk": MAX_TOPK,
            },
        )
        if stop == len(embeddings) or stop % (batch_size * 20) == 0:
            print(
                f"RETRIEVAL_PROGRESS path={path.name} rows={stop}/{len(embeddings)}",
                flush=True,
            )
    if (np.asarray(result) < 0).any():
        raise RuntimeError(f"Retrieval contains invalid IDs: {path}")
    return np.load(path, mmap_mode="r")


def probes_or_load(index, embeddings, path: Path, batch_size: int):
    import numpy as np

    expected = (len(embeddings), NPROBE)
    if path.exists():
        result = np.load(path, mmap_mode="r")
        if result.shape != expected:
            raise ValueError(f"Invalid probe cache {path}: {result.shape}")
        return result
    result = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=expected)
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        _, clusters = index.quantizer.search(
            np.ascontiguousarray(embeddings[start:stop]), NPROBE
        )
        result[start:stop] = clusters
    result.flush()
    return np.load(path, mmap_mode="r")


def nearest_other_or_load(
    embeddings, indices_path: Path, similarities_path: Path, batch_size: int
):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    expected = (len(embeddings),)
    if indices_path.exists() and similarities_path.exists():
        indices = np.load(indices_path, mmap_mode="r")
        similarities = np.load(similarities_path, mmap_mode="r")
        if indices.shape != expected or similarities.shape != expected:
            raise ValueError("Invalid nearest-neighbor cache")
        return indices, similarities

    torch.backends.cuda.matmul.allow_tf32 = False
    vectors = torch.from_numpy(np.asarray(embeddings)).to("cuda", dtype=torch.float32)
    vectors = functional.normalize(vectors, p=2, dim=1)
    indices = np.empty(len(embeddings), dtype=np.int64)
    similarities = np.empty(len(embeddings), dtype=np.float32)
    began = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            stop = min(start + batch_size, len(embeddings))
            matrix = vectors[start:stop] @ vectors.T
            local_rows = torch.arange(stop - start, device="cuda")
            global_rows = torch.arange(start, stop, device="cuda")
            matrix[local_rows, global_rows] = -float("inf")
            values, nearest = matrix.max(dim=1)
            indices[start:stop] = nearest.cpu().numpy()
            similarities[start:stop] = values.clamp_(-1.0, 1.0).cpu().numpy()
            del matrix, values, nearest
            if stop == len(embeddings) or stop % (batch_size * 10) == 0:
                print(
                    f"NEAREST_PROGRESS rows={stop}/{len(embeddings)} elapsed_seconds={time.perf_counter() - began:.1f}",
                    flush=True,
                )
    del vectors
    torch.cuda.empty_cache()
    np.save(indices_path, indices)
    np.save(similarities_path, similarities)
    return indices, similarities


def direct_document_overlap(doc_ids, nearest_indices, similarities) -> list[dict]:
    import numpy as np

    rows = []
    paired = np.asarray(doc_ids)[np.asarray(nearest_indices)]
    overlap_by_k = {}
    exact_by_k = {}
    for topk in TOPK_VALUES:
        current = np.asarray(doc_ids[:, :topk])
        neighbor = paired[:, :topk]
        overlap = (current[:, :, None] == neighbor[:, None, :]).any(axis=2).sum(axis=1)
        exact = np.all(np.sort(current, axis=1) == np.sort(neighbor, axis=1), axis=1)
        overlap_by_k[topk] = overlap
        exact_by_k[topk] = exact
    similarities = np.asarray(similarities)
    for threshold in THRESHOLDS:
        mask = similarities >= threshold
        count = int(mask.sum())
        row = {"similarity_threshold": threshold, "query_pair_count": count}
        for topk in TOPK_VALUES:
            overlap = overlap_by_k[topk][mask]
            exact = exact_by_k[topk][mask]
            row[f"top{topk}_mean_overlap_count"] = (
                float(overlap.mean()) if count else None
            )
            row[f"top{topk}_mean_overlap_ratio"] = (
                float(overlap.mean() / topk) if count else None
            )
            row[f"top{topk}_exact_set_match_rate"] = (
                float(exact.mean()) if count else None
            )
        rows.append(row)
    return rows


def doc_to_cluster_or_load(index, path: Path):
    import faiss
    import numpy as np

    expected = (int(index.ntotal),)
    if path.exists():
        mapping = np.load(path, mmap_mode="r")
        if mapping.shape != expected or mapping.dtype != np.int32:
            raise ValueError(
                f"Invalid doc-to-cluster cache: {mapping.shape}, {mapping.dtype}"
            )
        return mapping
    mapping = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=expected)
    mapping.fill(-1)
    for cluster in range(int(index.nlist)):
        size = int(index.invlists.list_size(cluster))
        if size:
            pointer = index.invlists.get_ids(cluster)
            ids = faiss.rev_swig_ptr(pointer, size)
            mapping[np.asarray(ids, dtype=np.int64)] = cluster
        if (cluster + 1) % 512 == 0:
            print(f"DOC_CLUSTER_PROGRESS lists={cluster + 1}/{index.nlist}", flush=True)
    mapping.flush()
    if (np.asarray(mapping) < 0).any():
        raise RuntimeError("Some document IDs were not assigned to an IVF cluster")
    return np.load(path, mmap_mode="r")


def probe_overlap_counts(probes, nearest_indices, chunk_size: int = 256):
    import numpy as np

    counts = np.empty(len(probes), dtype=np.int16)
    nearest_indices = np.asarray(nearest_indices)
    for start in range(0, len(probes), chunk_size):
        stop = min(start + chunk_size, len(probes))
        left = np.asarray(probes[start:stop])
        right = np.asarray(probes[nearest_indices[start:stop]])
        counts[start:stop] = (
            (left[:, :, None] == right[:, None, :]).any(axis=2).sum(axis=1)
        )
    return counts


def probe_rank_summary(doc_ids, probes, doc_to_cluster, topk: int) -> dict:
    import numpy as np

    document_clusters = np.asarray(doc_to_cluster[np.asarray(doc_ids[:, :topk])])
    matches = document_clusters[:, :, None] == np.asarray(probes)[:, None, :]
    if not matches.any(axis=2).all():
        missing = int((~matches.any(axis=2)).sum())
        raise RuntimeError(
            f"{missing} retrieved documents do not belong to the selected probe set"
        )
    ranks = matches.argmax(axis=2).astype(np.int16) + 1
    per_query_mean = ranks.mean(axis=1)
    return {
        "topk": topk,
        "global_mean_probe_rank": float(ranks.mean()),
        "global_p50_probe_rank": float(np.percentile(ranks, 50)),
        "global_p95_probe_rank": float(np.percentile(ranks, 95)),
        "global_p99_probe_rank": float(np.percentile(ranks, 99)),
        "p99_per_query_mean_probe_rank": float(np.percentile(per_query_mean, 99)),
        "max_probe_rank": int(ranks.max()),
    }


def main() -> None:
    args = parse_args()
    selected_cpus = configure_resources(args)

    import faiss
    import numpy as np
    import torch

    from mnemorag.retriever.encoder import Encoder

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(args.cpu_threads)
    print(f"INDEX_LOAD_START path={INDEX_PATH}", flush=True)
    index = faiss.read_index(str(INDEX_PATH))
    if not isinstance(index, faiss.IndexIVF) or int(index.nlist) != 4096:
        raise TypeError(
            f"Unexpected index: {type(index)!r}, nlist={getattr(index, 'nlist', None)}"
        )
    print(f"INDEX_LOAD_DONE ntotal={index.ntotal} nlist={index.nlist}", flush=True)

    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = Encoder(
                model_name="e5",
                model_path=str(MODEL_PATH),
                pooling_method="mean",
                max_length=128,
                use_fp16=True,
                instruction=None,
                silent=True,
            )
        return encoder

    # Panel (a): complete NQ train (79,168 queries).
    nq_questions = load_questions(NQ_TRAIN)
    if len(nq_questions) != 79168:
        raise ValueError(f"Expected 79,168 NQ train queries, got {len(nq_questions)}")
    nq_embedding_source = (
        PROJECT_ROOT / "results/background_motivation/cache/nq_query_embeddings.npy"
    )
    nq_doc_source = (
        PROJECT_ROOT
        / "results/background_motivation/cache/nq_nprobe128_top10_doc_ids.npy"
    )
    nq_embeddings = np.load(nq_embedding_source, mmap_mode="r")
    nq_doc_ids = np.load(nq_doc_source, mmap_mode="r")
    if nq_embeddings.shape != (79168, 1024) or nq_doc_ids.shape != (79168, 10):
        raise ValueError("Unexpected complete-NQ cache shape")
    nq_nearest, nq_similarities = nearest_other_or_load(
        nq_embeddings,
        cache / "nq_train_nearest_other_indices_exact_fp32.npy",
        cache / "nq_train_nearest_other_similarity_exact_fp32.npy",
        args.nn_batch_size,
    )
    panel_a_rows = direct_document_overlap(nq_doc_ids, nq_nearest, nq_similarities)
    write_csv(args.output_dir / "figure7a_nq_direct_document_overlap.csv", panel_a_rows)
    print("PANEL_A_DONE dataset=nq queries=79168", flush=True)

    doc_to_cluster = doc_to_cluster_or_load(
        index, cache / "wiki18_doc_to_ivf_cluster.npy"
    )
    panel_b_rows = []
    supplementary_rows = []
    dataset_manifest = {}
    for dataset, spec in EVAL_DATASETS.items():
        path = spec["path"]
        questions = load_questions(path)
        if len(questions) != spec["expected_queries"]:
            raise ValueError(
                f"{dataset}: expected {spec['expected_queries']} queries, got {len(questions)}"
            )
        dataset_manifest[dataset] = {
            "display": spec["display"],
            "path": str(path),
            "source_split": spec["source_split"],
            "query_count": len(questions),
            "sha256": sha256(path),
        }
        print(
            f"PANEL_B_DATASET_START name={dataset} queries={len(questions)}", flush=True
        )
        embeddings = encode_or_load(
            get_encoder(),
            questions,
            cache / f"{dataset}_test_query_embeddings.npy",
            args.encoding_batch_size,
        )
        doc_ids = retrieve_or_load(
            index,
            embeddings,
            cache / f"{dataset}_test_nprobe128_top10_doc_ids.npy",
            cache / f"{dataset}_test_nprobe128_top10_state.json",
            args.retrieval_batch_size,
        )
        probes = probes_or_load(
            index,
            embeddings,
            cache / f"{dataset}_test_top128_probe_clusters.npy",
            args.retrieval_batch_size,
        )
        nearest, similarities = nearest_other_or_load(
            embeddings,
            cache / f"{dataset}_test_nearest_other_indices_exact_fp32.npy",
            cache / f"{dataset}_test_nearest_other_similarity_exact_fp32.npy",
            args.nn_batch_size,
        )
        overlap = probe_overlap_counts(probes, nearest)
        rank_summaries = {
            topk: probe_rank_summary(doc_ids, probes, doc_to_cluster, topk)
            for topk in TOPK_VALUES
        }
        top3 = rank_summaries[3]
        panel_b_rows.append(
            {
                "dataset": dataset,
                "display_name": spec["display"],
                "query_count": len(questions),
                "nprobe": NPROBE,
                "nearest_pair_mean_similarity": float(np.asarray(similarities).mean()),
                "mean_probe_overlap_count": float(overlap.mean()),
                "mean_probe_overlap_ratio": float(overlap.mean() / NPROBE),
                "topk_for_rank": 3,
                "global_mean_probe_rank": top3["global_mean_probe_rank"],
                "p99_per_query_mean_probe_rank": top3["p99_per_query_mean_probe_rank"],
                "global_p99_probe_rank": top3["global_p99_probe_rank"],
            }
        )
        for topk in TOPK_VALUES:
            supplementary_rows.append(
                {
                    "dataset": dataset,
                    "display_name": spec["display"],
                    "query_count": len(questions),
                    **rank_summaries[topk],
                }
            )
        write_csv(
            args.output_dir / "figure7b_probe_overlap_and_top3_rank.csv", panel_b_rows
        )
        write_csv(
            args.output_dir / "figure7_supplementary_topk_probe_rank.csv",
            supplementary_rows,
        )
        print(
            f"PANEL_B_DATASET_DONE name={dataset} mean_overlap={overlap.mean():.6f} "
            f"top3_mean_rank={top3['global_mean_probe_rank']:.6f} "
            f"top3_p99_query_mean_rank={top3['p99_per_query_mean_probe_rank']:.6f}",
            flush=True,
        )

    manifest = {
        "status": "complete",
        "figure": 7,
        "definitions": {
            "nearest_pair": "For every query, globally most cosine-similar other query; directed, one pair per query; self excluded.",
            "panel_a_overlap": "Direct document-ID set intersection |R_k(q) intersect R_k(q_nearest)|.",
            "panel_b_overlap": "Probe-cluster set intersection |C_128(q) intersect C_128(q_nearest)|.",
            "panel_b_rank": "One-based position of each retrieved document's IVF cluster in that query's ordered Top-128 probe list.",
            "panel_b_p99": "99th percentile across per-query mean Top-k probe rank, matching the original helper implementation.",
        },
        "panel_a": {
            "dataset": "NQ",
            "split": "train (largest complete split)",
            "query_count": len(nq_questions),
            "path": str(NQ_TRAIN),
            "sha256": sha256(NQ_TRAIN),
            "topk_values": list(TOPK_VALUES),
            "similarity_thresholds": list(THRESHOLDS),
            "embedding_cache": str(nq_embedding_source),
            "retrieval_cache": str(nq_doc_source),
        },
        "panel_b": {
            "datasets": dataset_manifest,
            "nprobe": NPROBE,
            "primary_rank_topk": 3,
            "audited_rank_topk": list(TOPK_VALUES),
        },
        "index": {
            "path": str(INDEX_PATH),
            "size_bytes": INDEX_PATH.stat().st_size,
            "ntotal": int(index.ntotal),
            "nlist": int(index.nlist),
        },
        "embedding_model": str(MODEL_PATH),
        "resources": {
            "physical_gpu_id": str(args.gpu_id),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "cpu_threads": args.cpu_threads,
            "cpu_affinity": selected_cpus,
            "faiss_version": faiss.__version__,
            "torch_version": torch.__version__,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"FIGURE7_EXACT_OVERLAP_OK output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
