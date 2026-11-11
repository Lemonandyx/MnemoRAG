#!/usr/bin/env python3
"""Run one isolated retrieval-only point for MnemoRAG and cache baselines."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .benchmark_point import enforce_fixed_64_core_contract, resource_preflight
except ImportError:
    from benchmark_point import enforce_fixed_64_core_contract, resource_preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_DIR = ASSET_ROOT / "qa/datasets/nq"
INDEX_PATH = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
CORPUS_PATH = ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
MODEL_PATH = ASSET_ROOT / "models/e5-large-v2"
BASELINE_ARTIFACTS = Path(
    os.environ.get(
        "MNEMORAG_BASELINE_ARTIFACTS",
        PROJECT_ROOT / "results/retrieval_baseline_artifacts",
    )
).resolve()
QV_SAMPLE_PATH = BASELINE_ARTIFACTS / "wiki18_e5_qvcache_pca_sample.f32.bin"
QV_PCA_PREFIX = BASELINE_ARTIFACTS / "wiki18_e5_qvcache"
DOC_VECTORS = Path(
    os.environ.get(
        "MNEMORAG_VECTOR_PATH",
        PROJECT_ROOT / "assets/wiki18_2100w/vectors.f16.npy",
    )
).resolve()
PROXIMIPY_LIBRARY = Path(
    os.environ.get(
        "PROXIMIPY_LIBRARY",
        PROJECT_ROOT / "third_party/Proximity/bindings/target/release/libproximipy.so",
    )
)
HNSWLIB_LIBRARY = Path(
    os.environ.get(
        "HNSWLIB_LIBRARY",
        PROJECT_ROOT / "third_party/hnswlib/hnswlib.so",
    )
)


def load_proximipy():
    try:
        import proximipy

        return proximipy
    except ModuleNotFoundError:
        if not PROXIMIPY_LIBRARY.exists():
            raise ModuleNotFoundError(
                f"proximipy is neither installed nor available at {PROXIMIPY_LIBRARY}"
            )
        spec = importlib.util.spec_from_file_location("proximipy", PROXIMIPY_LIBRARY)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {PROXIMIPY_LIBRARY}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["proximipy"] = module
        spec.loader.exec_module(module)
        return module


def load_hnswlib():
    try:
        import hnswlib

        return hnswlib
    except ModuleNotFoundError:
        if not HNSWLIB_LIBRARY.exists():
            raise ModuleNotFoundError(
                f"hnswlib is neither installed nor available at {HNSWLIB_LIBRARY}"
            )
        spec = importlib.util.spec_from_file_location("hnswlib", HNSWLIB_LIBRARY)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {HNSWLIB_LIBRARY}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["hnswlib"] = module
        spec.loader.exec_module(module)
        return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=("mnemorag", "faiss", "proximity", "qvcache", "has"),
        required=True,
    )
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--dataset-split", choices=("train", "test"), default="test")
    parser.add_argument("--nprobe", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument(
        "--workload-mode",
        choices=("iid_unique", "exact_repeat", "mixed_repeat"),
        default="iid_unique",
    )
    parser.add_argument(
        "--unique-queries",
        type=int,
        default=None,
        help=(
            "Unique-query count for exact_repeat, or warm exact-hit set size "
            "for mixed_repeat."
        ),
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--batch-wait-ms", type=float, default=100.0)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--faiss-omp-threads", type=int, default=1)
    parser.add_argument("--faiss-search-workers", type=int, default=64)
    parser.add_argument(
        "--direct-faiss-mode",
        choices=("native_batch", "query_parallel"),
        default="query_parallel",
    )
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-preflight-cpu-busy-percent", type=float, default=20.0)
    parser.add_argument(
        "--max-preflight-cpu-max-busy-percent", type=float, default=50.0
    )
    parser.add_argument("--max-preflight-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-preflight-gpu-util-percent", type=float, default=10.0)
    parser.add_argument("--max-preflight-load-per-cpu", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cache-capacity", type=int, default=80_000)
    parser.add_argument("--similarity-t-reuse", type=float, default=0.95)
    parser.add_argument("--similarity-t-guided", type=float, default=0.70)
    parser.add_argument(
        "--exact-query-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--has-cache-capacity", type=int, default=2_048)
    parser.add_argument("--proximity-tau", type=float, default=0.48)
    parser.add_argument("--has-threshold", type=float, default=0.6)
    parser.add_argument(
        "--qvcache-regional-theta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--qvcache-lazy-theta-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--qvcache-metric", choices=("cosine", "l2"), default="cosine")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation/retrieval_cache_scaling",
    )
    return parser.parse_args()


def configure(args: argparse.Namespace) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    args.cpu_topology = enforce_fixed_64_core_contract(args, available)
    selected = available
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.effective_faiss_omp_threads = args.faiss_omp_threads
    args.effective_faiss_search_workers = args.faiss_search_workers
    os.environ["MNEMORAG_FAISS_OMP_THREADS"] = str(args.effective_faiss_omp_threads)
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.effective_faiss_omp_threads)
    os.environ["MNEMORAG_FAISS_SEARCH_WORKERS"] = str(
        args.effective_faiss_search_workers
    )
    os.environ["MNEMORAG_FAISS_BATCH_FULL"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MNEMORAG_FAISS_BATCH_FULL"] = (
        "1" if args.direct_faiss_mode == "native_batch" else "0"
    )
    args.resource_preflight = resource_preflight(args, selected)
    print(
        "Retrieval-cache resource preflight: "
        + json.dumps(args.resource_preflight, sort_keys=True),
        flush=True,
    )
    return selected


def load_questions(
    count: int,
    seed: int,
    split: str,
    workload_mode: str = "iid_unique",
    unique_queries: int | None = None,
) -> tuple[list[str], list[int], int, Path]:
    dataset_path = DATASET_DIR / f"{split}.jsonl"
    rows = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    requested_unique = count
    if workload_mode == "exact_repeat":
        requested_unique = unique_queries or max(1, count // 4)
        if requested_unique > count:
            raise ValueError("--unique-queries cannot exceed --num-requests")
    elif workload_mode == "mixed_repeat":
        warm_unique = unique_queries or max(1, count // 4)
        if warm_unique < 1 or count - warm_unique < 2:
            raise ValueError(
                "mixed_repeat requires at least one warm query and one repeat/new pair"
            )
        requested_unique = warm_unique + (count - warm_unique) // 2
    elif unique_queries is not None and unique_queries != count:
        raise ValueError("--unique-queries applies only to repeat workloads")
    if requested_unique > len(rows):
        raise ValueError(
            f"Requested {requested_unique} unique rows from {len(rows)} available"
        )
    rng = random.Random(seed)
    base_indices = sorted(rng.sample(range(len(rows)), requested_unique))
    if workload_mode == "mixed_repeat":
        warm_unique = unique_queries or max(1, count // 4)
        warm_indices = base_indices[:warm_unique]
        cold_indices = base_indices[warm_unique:]
        indices = list(warm_indices)
        for offset in range(count - warm_unique):
            if offset % 2 == 0:
                indices.append(warm_indices[(offset // 2) % warm_unique])
            else:
                indices.append(cold_indices[offset // 2])
    else:
        indices = [base_indices[index % requested_unique] for index in range(count)]
    return (
        [rows[index]["question"] for index in indices],
        indices,
        len(rows),
        dataset_path,
    )


class DocumentVectorStore:
    def __init__(self) -> None:
        self.vectors = np.load(DOC_VECTORS, mmap_mode="r")
        if self.vectors.shape[1] != 1024:
            raise RuntimeError("Unexpected document embedding dimension")

    def fetch(self, ids: Iterable[int]) -> list[np.ndarray]:
        indices = np.asarray(list(ids), dtype=np.int64)
        return [row for row in np.asarray(self.vectors[indices], dtype=np.float32)]

    def fetch_array(self, ids: Iterable[int]) -> np.ndarray:
        indices = np.asarray(list(ids), dtype=np.int64)
        return np.asarray(self.vectors[indices], dtype=np.float32)


class FaissBackend:
    """QVCache custom backend over the campaign's common IVF index."""

    def __init__(
        self, index, vectors: DocumentVectorStore, metric: str = "cosine"
    ) -> None:
        self.index = index
        self.vectors = vectors
        self.metric = metric

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(
            np.asarray(query, dtype=np.float32).reshape(1, -1), k
        )
        if self.metric == "cosine":
            distances = 1.0 - scores[0]
        elif self.metric == "l2":
            # E5 document and query vectors are unit normalized.
            distances = 2.0 - 2.0 * scores[0]
        else:
            raise AssertionError(self.metric)
        return ids[0].astype(np.uint32), distances.astype(np.float32)

    def fetch_vectors_by_ids(self, ids: list[int]) -> list[np.ndarray]:
        return self.vectors.fetch(ids)


class DirectEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        import faiss

        from mnemorag.retriever.encoder import Encoder
        from mnemorag.retriever.utils import load_corpus

        self.args = args
        self.faiss = faiss
        self.system = args.system
        self.encoder = Encoder(
            model_name="e5",
            model_path=str(MODEL_PATH),
            pooling_method="mean",
            max_length=128,
            use_fp16=True,
            instruction=None,
            silent=True,
        )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.index.nprobe = args.nprobe
        self.corpus = load_corpus(str(CORPUS_PATH))
        self.vector_store = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.fuzzy_hits = 0
        self.cache = None
        self.qvcache = None
        self.has_index = None
        self.has_cache = None
        self.has_query_id = 0
        self.search_executor = (
            ThreadPoolExecutor(
                max_workers=args.effective_faiss_search_workers,
                thread_name_prefix="retrieval-cache-faiss",
            )
            if args.direct_faiss_mode == "query_parallel"
            and self.system in {"faiss", "proximity"}
            else None
        )
        faiss.omp_set_num_threads(args.effective_faiss_omp_threads)
        if self.system == "proximity":
            proximipy = load_proximipy()
            self.cache = proximipy.FifoCache(args.cache_capacity)
        elif self.system == "qvcache":
            self._init_qvcache()
        elif self.system == "has":
            self._init_has()

    def _init_qvcache(self) -> None:
        qvcache_root = Path(
            os.environ.get("QVCACHE_ROOT", PROJECT_ROOT / "third_party/QVCache")
        ).resolve()
        sys.path.insert(0, str(qvcache_root / "python"))
        import qvcache as qvc

        if not QV_SAMPLE_PATH.exists():
            raise FileNotFoundError(QV_SAMPLE_PATH)
        self.vector_store = DocumentVectorStore()
        backend = FaissBackend(
            self.index, self.vector_store, metric=self.args.qvcache_metric
        )
        metric = (
            qvc.Metric.COSINE if self.args.qvcache_metric == "cosine" else qvc.Metric.L2
        )
        self.qvcache = qvc.QVCache(
            data_path=str(QV_SAMPLE_PATH),
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
            deviation_factor=0.25,
            memory_index_max_points=self.args.cache_capacity,
            beamwidth=2,
            use_regional_theta=self.args.qvcache_regional_theta,
            pca_dim=16,
            buckets_per_dim=8,
            max_regions=1_000_000,
            n_async_insert_threads=4,
            lazy_theta_updates=self.args.qvcache_lazy_theta_updates,
            number_of_mini_indexes=4,
            search_mini_indexes_in_parallel=False,
            max_search_threads=32,
            metric=metric,
            backend=backend,
        )
        self.qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
        self.qvcache.enable_adaptive_strategy(True)
        self.qvcache.set_hit_ratio_window_size(100)
        self.qvcache.set_hit_ratio_threshold(0.90)

    def _init_has(self) -> None:
        load_hnswlib()
        has_root = Path(
            os.environ.get("HAS_ROOT", PROJECT_ROOT / "third_party/HaS")
        ).resolve()
        sys.path.insert(0, str(has_root / "src"))
        from SpeculativeRetrievalService.HaS import CacheChannel
        from utils.index import HNSWLibIndexerV2

        self.vector_store = DocumentVectorStore()
        # Controlled comparison: HaS's fuzzy channel and fallback share the
        # exact same IVF4096,Flat corpus index as every other system.  Only the
        # fuzzy pass lowers nprobe to 64; no independent IVFPQ/PQ artifact is
        # used in the main figure.
        self.has_index = self.index
        self.has_cache = CacheChannel()
        self.has_cache.index = HNSWLibIndexerV2(vector_sz=1024)
        # The official implementation reserves max_size * 30 HNSW vector
        # slots.  With top-k=5 and 1024 requests, 2048 query entries already
        # cover the complete trace without eviction while avoiding a needless
        # 3-million-vector preallocation.
        self.has_cache.reset(self.args.has_cache_capacity)

    @staticmethod
    def _materialize(corpus, ids: np.ndarray) -> None:
        # Match the existing Figure 13 path: retrieval completes only once the
        # selected document records are available to the caller.
        flat = ids.reshape(-1).tolist()
        _ = [corpus[index] for index in flat]

    def close(self) -> None:
        if self.search_executor is not None:
            self.search_executor.shutdown(wait=True, cancel_futures=True)

    def _search_faiss(
        self,
        embeddings: np.ndarray,
        k: int,
        nprobe: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        probe_count = self.args.nprobe if nprobe is None else int(nprobe)
        if self.search_executor is None or len(embeddings) <= 1:
            params = self.faiss.SearchParametersIVF()
            params.nprobe = probe_count
            return self.index.search(embeddings, k, params=params)

        def search_one(row: np.ndarray):
            params = self.faiss.SearchParametersIVF()
            params.nprobe = probe_count
            return self.index.search(row.reshape(1, -1), k, params=params)

        outputs = list(self.search_executor.map(search_one, embeddings))
        scores = np.concatenate([output[0] for output in outputs], axis=0)
        ids = np.concatenate([output[1] for output in outputs], axis=0)
        return scores, ids

    def search(self, questions: list[str]) -> tuple[np.ndarray, dict]:
        embeddings = self.encoder.encode(
            questions, batch_size=self.args.retrieval_batch_size, is_query=True
        )
        if self.system == "faiss":
            _, ids = self._search_faiss(embeddings, self.args.topk)
        elif self.system == "proximity":
            ids = self._search_proximity(embeddings)
        elif self.system == "qvcache":
            ids = self._search_qvcache(embeddings)
        elif self.system == "has":
            ids = self._search_has(embeddings)
        else:
            raise AssertionError(self.system)
        self._materialize(self.corpus, ids)
        stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "fuzzy_hits": self.fuzzy_hits,
        }
        if self.qvcache is not None:
            stats.update(
                {
                    "memory_active_vectors": self.qvcache.get_number_of_vectors_in_memory_index(),
                    "memory_max_points": self.qvcache.get_number_of_max_points_in_memory_index(),
                    "pca_active_regions": self.qvcache.get_number_of_active_pca_regions(),
                    "mini_index_vectors": [
                        self.qvcache.get_index_vector_count(index)
                        for index in range(self.qvcache.get_number_of_mini_indexes())
                    ],
                }
            )
        return ids, stats

    def _search_proximity(self, embeddings: np.ndarray) -> np.ndarray:
        found = self.cache.batch_find(embeddings.tolist())
        ids = np.empty((len(embeddings), self.args.topk), dtype=np.int64)
        misses = [index for index, value in enumerate(found) if value is None]
        if misses:
            scores, miss_ids = self._search_faiss(embeddings[misses], self.args.topk)
            for local, index in enumerate(misses):
                ids[index] = miss_ids[local]
                value = (miss_ids[local].tolist(), scores[local].tolist())
                self.cache.insert(
                    embeddings[index].tolist(), value, self.args.proximity_tau
                )
        for index, value in enumerate(found):
            if value is not None:
                ids[index] = np.asarray(value[0], dtype=np.int64)
        self.cache_hits += len(embeddings) - len(misses)
        self.cache_misses += len(misses)
        return ids

    def _search_qvcache(self, embeddings: np.ndarray) -> np.ndarray:
        ids = np.empty((len(embeddings), self.args.topk), dtype=np.int64)
        for index, embedding in enumerate(embeddings):
            hit, result_ids, _ = self.qvcache.search(embedding, self.args.topk)
            one_based_ids = np.asarray(result_ids, dtype=np.int64)
            if np.any(one_based_ids < 1) or np.any(one_based_ids > len(self.corpus)):
                raise RuntimeError("QVCache returned an invalid one-based document ID")
            # QVCache reserves zero internally and increments backend IDs on misses.
            ids[index] = one_based_ids - 1
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
        return ids

    def _search_has(self, embeddings: np.ndarray) -> np.ndarray:
        ids = np.empty((len(embeddings), self.args.topk), dtype=np.int64)
        for index, embedding in enumerate(embeddings):
            self.has_index.nprobe = 64
            fuzzy_scores, fuzzy_ids = self.has_index.search(
                embedding.reshape(1, -1), self.args.topk * 5
            )
            fuzzy_vectors = self.vector_store.fetch_array(fuzzy_ids[0])
            exact_scores = fuzzy_vectors @ embedding
            order = np.argsort(-exact_scores)[: self.args.topk]
            fuzzy_docs = [
                {"id": int(fuzzy_ids[0, pos]), "score": float(exact_scores[pos])}
                for pos in order
            ]
            cached_docs = self.has_cache.search(embedding, top_n=self.args.topk)
            merged = {}
            for doc in fuzzy_docs + cached_docs:
                merged[doc["id"]] = max(merged.get(doc["id"], -np.inf), doc["score"])
            candidate_ids = [
                doc_id
                for doc_id, _ in sorted(
                    merged.items(), key=lambda item: item[1], reverse=True
                )[: self.args.topk]
            ]
            previous, _ = self.has_cache.is_similar_query_exist(
                candidate_ids, self.args.has_threshold
            )
            if previous is not None:
                result = candidate_ids
                self.cache_hits += 1
                self.fuzzy_hits += 1
            else:
                self.index.nprobe = self.args.nprobe
                _, full_ids = self.index.search(
                    embedding.reshape(1, -1), self.args.topk
                )
                result = full_ids[0].tolist()
                doc_vectors = self.vector_store.fetch_array(result)
                self.has_cache.add(
                    {"id": self.has_query_id}, np.asarray(result), doc_vectors
                )
                self.has_query_id += 1
                self.cache_misses += 1
            ids[index] = np.asarray(result, dtype=np.int64)
        return ids


class MnemoEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        from mnemorag.config import Config
        from mnemorag.pipeline.utils import QueryPool
        from mnemorag.utils import get_retriever

        config = Config(
            config_dict={
                "split": ["test"],
                "dataset_name": "nq",
                "data_dir": str(ASSET_ROOT / "qa/datasets"),
                "index_path": str(INDEX_PATH),
                "corpus_path": str(CORPUS_PATH),
                "model2path": {"e5": str(MODEL_PATH)},
                "retrieval_method": "e5",
                "retrieval_topk": args.topk,
                "retrieval_batch_size": args.retrieval_batch_size,
                "ivf_nprobe": args.nprobe,
                "enable_similarity_aware": True,
                "similarity_t_reuse": args.similarity_t_reuse,
                "similarity_t_guided": args.similarity_t_guided,
                "gpu_id": args.gpu_id,
                "gpu_num": 1,
                "cpu_threads": args.cpu_threads,
                "query_base_capacity": args.cache_capacity,
                "query_base_dtype": "float16",
                "query_base_policy": "lru",
                "seed": args.seed,
                "disable_save": True,
                "silent_retrieval": True,
            }
        )
        self.args = args
        self.retriever = get_retriever(config)
        self.query_pool = QueryPool(
            device="cuda",
            max_size=args.cache_capacity,
            enable_exact_lookup=args.exact_query_cache,
        )
        self.counts = {"exact": 0, "reuse": 0, "guided": 0, "full": 0}

    def search(self, questions: list[str]) -> tuple[np.ndarray, dict]:
        _, _, ids = self.retriever.batch_search(
            questions,
            num=self.args.topk,
            query_pool=self.query_pool,
            return_score=True,
            return_idx=True,
        )
        counts = getattr(self.retriever, "last_similarity_action_counts", {})
        for name in self.counts:
            self.counts[name] += int(counts.get(name, 0))
        stats = dict(self.counts)
        profile = self.retriever.retrieval_profile()
        stats["adaptive_reuse_first"] = profile["adaptive_reuse_first"]
        stats["exact_query_fast_path"] = profile["exact_query_fast_path"]
        return np.asarray(ids, dtype=np.int64), stats

    def close(self) -> None:
        self.retriever.close()


async def run(args: argparse.Namespace, selected_cpus: list[int]) -> dict:
    questions, sample_indices, source_count, dataset_path = load_questions(
        args.num_requests,
        args.seed,
        args.dataset_split,
        args.workload_mode,
        args.unique_queries,
    )
    engine = MnemoEngine(args) if args.system == "mnemorag" else DirectEngine(args)
    rng = np.random.default_rng(args.seed)
    arrivals = np.cumsum(rng.exponential(1.0 / args.arrival_rate, args.num_requests))
    queue: asyncio.Queue = asyncio.Queue()
    finish = np.zeros(args.num_requests, dtype=float)
    returned_ids = np.full((args.num_requests, args.topk), -1, dtype=np.int64)
    service_times: list[float] = []
    batch_sizes: list[int] = []
    last_stats: dict = {}
    origin = time.perf_counter()

    async def producer() -> None:
        for index, (question, target) in enumerate(zip(questions, arrivals)):
            delay = float(target) - (time.perf_counter() - origin)
            if delay > 0:
                await asyncio.sleep(delay)
            await queue.put((index, question))
        await queue.put(None)

    async def consumer() -> None:
        nonlocal last_stats
        input_done = False
        while not input_done:
            first = await queue.get()
            if first is None:
                break
            batch = [first]
            deadline = time.perf_counter() + args.batch_wait_ms / 1000.0
            while len(batch) < args.retrieval_batch_size and not input_done:
                if not queue.empty():
                    item = queue.get_nowait()
                else:
                    timeout = deadline - time.perf_counter()
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                if item is None:
                    input_done = True
                    break
                batch.append(item)
            indices, batch_questions = zip(*batch)
            started = time.perf_counter()
            ids, last_stats = await asyncio.to_thread(
                engine.search, list(batch_questions)
            )
            service_times.append(time.perf_counter() - started)
            batch_sizes.append(len(batch))
            completed = time.perf_counter() - origin
            for local, index in enumerate(indices):
                returned_ids[index] = ids[local]
                finish[index] = completed

    try:
        await asyncio.gather(producer(), consumer())
    finally:
        engine.close()
    elapsed = time.perf_counter() - origin
    latency = finish - arrivals
    point_name = (
        f"{args.system}_np{args.nprobe}_rps{args.arrival_rate:g}_"
        f"k{args.topk}_n{args.num_requests}_{args.workload_mode}_nq{args.dataset_split}"
    )
    point_dir = args.output_dir / point_name
    point_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        point_dir / "requests.npz",
        sample_indices=np.asarray(sample_indices),
        arrival_offset_s=arrivals,
        finish_offset_s=finish,
        retrieval_latency_s=latency,
        returned_doc_ids=returned_ids,
    )
    summary = {
        "status": "complete",
        "point_name": point_name,
        "system": args.system,
        "dataset": f"nq_{args.dataset_split}",
        "dataset_path": str(dataset_path),
        "source_queries": source_count,
        "sample_indices": sample_indices,
        "seed": args.seed,
        "num_requests": args.num_requests,
        "workload_mode": args.workload_mode,
        "unique_queries": len(set(sample_indices)),
        "request_repetition_factor": args.num_requests / len(set(sample_indices)),
        "arrival_process": "Poisson",
        "arrival_rate_rps": args.arrival_rate,
        "nprobe": args.nprobe,
        "topk": args.topk,
        "retrieval_batch_size": args.retrieval_batch_size,
        "batch_wait_ms": args.batch_wait_ms,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "cpu_topology": args.cpu_topology,
        "resource_preflight": args.resource_preflight,
        "direct_faiss_mode": args.direct_faiss_mode,
        "faiss_omp_threads": args.effective_faiss_omp_threads,
        "faiss_search_workers": args.effective_faiss_search_workers,
        "physical_gpu_id": args.gpu_id,
        "wall_time_s": elapsed,
        "throughput_qps": args.num_requests / elapsed,
        "latency_mean_s": float(np.mean(latency)),
        "latency_p50_s": float(np.percentile(latency, 50)),
        "latency_p95_s": float(np.percentile(latency, 95)),
        "latency_p99_s": float(np.percentile(latency, 99)),
        "service_batch_time_mean_s": statistics.fmean(service_times),
        "batch_size_mean": statistics.fmean(batch_sizes),
        "batch_size_max": max(batch_sizes),
        "system_stats": last_stats,
        "cache_capacity": args.cache_capacity,
        "similarity_t_reuse": (
            args.similarity_t_reuse if args.system == "mnemorag" else None
        ),
        "similarity_t_guided": (
            args.similarity_t_guided if args.system == "mnemorag" else None
        ),
        "exact_query_cache": (
            args.exact_query_cache if args.system == "mnemorag" else None
        ),
        "proximity_tau": args.proximity_tau if args.system == "proximity" else None,
        "has_threshold": args.has_threshold if args.system == "has" else None,
        "has_cache_capacity": (
            args.has_cache_capacity if args.system == "has" else None
        ),
        "has_fuzzy_index": (
            "shared IVF4096,Flat with nprobe=64" if args.system == "has" else None
        ),
        "qvcache_regional_theta": (
            args.qvcache_regional_theta if args.system == "qvcache" else None
        ),
        "qvcache_lazy_theta_updates": (
            args.qvcache_lazy_theta_updates if args.system == "qvcache" else None
        ),
        "qvcache_metric": args.qvcache_metric if args.system == "qvcache" else None,
        "backend_index": "shared E5 IVF4096,Flat",
        "cache_isolation": "fresh process and empty cache per point",
        "metric_scope": "arrival-to-materialized-retrieval-result; no LLM",
    }
    temporary = point_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(point_dir / "summary.json")
    print("RETRIEVAL_CACHE_POINT_COMPLETE " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    selected = configure(args)
    asyncio.run(run(args, selected))


if __name__ == "__main__":
    main()
