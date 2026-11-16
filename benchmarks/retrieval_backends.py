"""Retrieval backends for comparison experiments."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

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
    """QVCache backend over the shared IVF index."""

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
        # Use the shared IVF index for HaS's fuzzy search and fallback.
        self.has_index = self.index
        self.has_cache = CacheChannel()
        self.has_cache.index = HNSWLibIndexerV2(vector_sz=1024)
        # Limit HNSW preallocation while retaining the full request trace.
        self.has_cache.reset(self.args.has_cache_capacity)

    @staticmethod
    def _materialize(corpus, ids: np.ndarray) -> None:
        # Include document loading in retrieval latency.
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
                "dataset_name": getattr(args, "dataset", "nq"),
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
