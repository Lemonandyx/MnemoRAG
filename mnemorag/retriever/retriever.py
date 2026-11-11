# ruff: noqa: E402

import json
import os

_DEFAULT_CPU_THREADS = "1"
faiss_omp_threads = int(
    os.environ.get("MNEMORAG_FAISS_OMP_THREADS", _DEFAULT_CPU_THREADS)
)
if faiss_omp_threads < 1:
    raise ValueError("MNEMORAG_FAISS_OMP_THREADS must be positive")
for variable in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, str(faiss_omp_threads))
import time
import uuid
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import functools
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import faiss
import numpy as np
import torch

from mnemorag.pipeline.utils import QueryPool, QueryState
from mnemorag.retriever.encoder import Encoder
from mnemorag.retriever.utils import (
    adaptive_probe_budget,
    convert_numpy,
    load_corpus,
    load_docs,
)
from mnemorag.utils import get_device, get_reranker

MNEMORAG_T_REUSE = 0.95
MNEMORAG_T_GUIDED = 0.85
_THRESHOLD_OVERRIDE_ENV = "MNEMORAG_ALLOW_THRESHOLD_OVERRIDE"


def _resolve_similarity_thresholds(config):
    t_reuse = MNEMORAG_T_REUSE
    t_guided = MNEMORAG_T_GUIDED
    if os.environ.get(_THRESHOLD_OVERRIDE_ENV) == "1":
        config_t_reuse = config["similarity_t_reuse"]
        config_t_guided = config["similarity_t_guided"]
        t_reuse = float(
            os.environ.get("MNEMORAG_T_REUSE") or config_t_reuse or MNEMORAG_T_REUSE
        )
        t_guided = float(
            os.environ.get("MNEMORAG_T_GUIDED") or config_t_guided or MNEMORAG_T_GUIDED
        )
    if t_guided > t_reuse:
        raise ValueError(f"T_GUIDED ({t_guided}) must be <= T_REUSE ({t_reuse})")
    return t_reuse, t_guided


def _merge_guided_probe_lists(
    query_probes,
    historical_clusters,
    *,
    allow_historical_expansion: bool,
):
    """Preserve query-list order and optionally append novel history lists."""
    merged, seen = [], set()
    for cluster in query_probes:
        cluster = int(cluster)
        if cluster not in seen:
            merged.append(cluster)
            seen.add(cluster)
    if allow_historical_expansion:
        for cluster in historical_clusters:
            cluster = int(cluster)
            if cluster not in seen:
                merged.append(cluster)
                seen.add(cluster)
    return merged


print("Faiss OMP threads before override:", faiss.omp_get_max_threads())
faiss.omp_set_num_threads(faiss_omp_threads)
print("Faiss OMP threads configured:", faiss_omp_threads)

if get_device() == "cpu":
    faiss.omp_set_num_threads(faiss_omp_threads)


def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(
        self,
        query=None,
        num=None,
        query_pool=None,
        return_score=False,
        return_idx=False,
    ):
        if num is None:
            num = self.topk
        single_query = isinstance(query, str)
        if self.use_cache:
            if single_query:
                new_query_list = [query]
            else:
                new_query_list = query

            no_cache_query = []
            cache_results = []
            for new_query in new_query_list:
                if new_query in self.cache:
                    cache_res = self.cache[new_query]
                    if len(cache_res) < num:
                        warnings.warn(
                            f"The number of cached retrieval results is less than topk ({num})"
                        )
                    cache_res = cache_res[:num]
                    docs, doc_scores, doc_idxs = [], [], []
                    for cached_item in cache_res:
                        item = dict(cached_item)
                        if "_retrieval_score" in item:
                            score = item.pop("_retrieval_score")
                        else:
                            score = item.pop("score", None)
                        doc_idx = item.pop("_retrieval_idx", None)
                        if doc_idx is None:
                            corpus_id = item.get("id")
                            if str(corpus_id).lstrip("-").isdigit():
                                doc_idx = int(corpus_id)
                        if score is None:
                            raise ValueError(
                                "Cached retrieval entry has no score; regenerate "
                                "the retrieval cache"
                            )
                        if doc_idx is None and return_idx:
                            raise ValueError(
                                "Cached retrieval entry has no Faiss document index; "
                                "regenerate the cache with the current format"
                            )
                        docs.append(item)
                        doc_scores.append(score)
                        doc_idxs.append(doc_idx)
                    cache_results.append((docs, doc_scores, doc_idxs))
                else:
                    cache_results.append(None)
                    no_cache_query.append(new_query)

            if no_cache_query != []:
                # use batch search without decorator
                (
                    no_cache_results,
                    no_cache_scores,
                    no_cache_idxs,
                ) = self._batch_search_with_rerank(
                    query=no_cache_query,
                    num=num,
                    query_pool=query_pool,
                    return_score=True,
                    return_idx=True,
                )
                no_cache_idx = 0
                for idx, res in enumerate(cache_results):
                    if res is None:
                        assert new_query_list[idx] == no_cache_query[no_cache_idx]
                        cache_results[idx] = (
                            no_cache_results[no_cache_idx],
                            no_cache_scores[no_cache_idx],
                            no_cache_idxs[no_cache_idx],
                        )
                        no_cache_idx += 1

            results, scores, idxs = (
                [t[0] for t in cache_results],
                [t[1] for t in cache_results],
                [t[2] for t in cache_results],
            )

        else:
            results, scores, idxs = func(
                self,
                query=query,
                query_pool=query_pool,
                num=num,
                return_score=True,
                return_idx=True,
            )
        if self.save_cache:
            save_queries = [query] if single_query else query
            save_results, save_scores, save_idxs = results, scores, idxs
            if single_query and "batch" not in func.__name__ and not self.use_cache:
                save_results = [save_results]
                save_scores = [save_scores]
                save_idxs = [save_idxs]
            for new_query, doc_items, doc_scores, doc_idxs in zip(
                save_queries, save_results, save_scores, save_idxs
            ):
                cached_items = []
                for item, score, doc_idx in zip(doc_items, doc_scores, doc_idxs):
                    cached_item = dict(item)
                    cached_item["_retrieval_score"] = score
                    cached_item["_retrieval_idx"] = int(doc_idx)
                    cached_items.append(cached_item)
                self.cache[new_query] = cached_items

        if self.use_cache and single_query and "batch" not in func.__name__:
            results, scores, idxs = results[0], scores[0], idxs[0]

        if return_idx:
            return results, scores, idxs
        if return_score:
            return results, scores
        else:
            return results

    return wrapper


def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(
        self, query, num=None, query_pool=None, return_score=False, return_idx=False
    ):
        results, scores, idxs = func(
            self,
            query=query,
            num=num,
            query_pool=query_pool,
            return_score=True,
            return_idx=True,
        )
        if self.use_reranker:
            original_results = results
            original_idxs = idxs
            results, scores = self.reranker.rerank(query, results)
            if "batch" not in func.__name__:
                results = results[0]
                scores = scores[0]
                original_results = [original_results]
                original_idxs = [original_idxs]
                reranked_batches = [results]
            else:
                reranked_batches = results

            aligned_idxs = []
            for source_docs, source_idxs, reranked_docs in zip(
                original_results, original_idxs, reranked_batches
            ):
                remaining = list(range(len(source_docs)))
                reordered = []
                for reranked_doc in reranked_docs:
                    match = next(
                        (pos for pos in remaining if source_docs[pos] is reranked_doc),
                        None,
                    )
                    if match is None and isinstance(reranked_doc, dict):
                        reranked_id = reranked_doc.get("id")
                        if reranked_id is not None:
                            match = next(
                                (
                                    pos
                                    for pos in remaining
                                    if isinstance(source_docs[pos], dict)
                                    and source_docs[pos].get("id") == reranked_id
                                ),
                                None,
                            )
                    if match is None:
                        match = next(
                            (
                                pos
                                for pos in remaining
                                if source_docs[pos] == reranked_doc
                            ),
                            None,
                        )
                    if match is None:
                        raise ValueError(
                            "Reranker returned a document that is absent from "
                            "the retrieval candidates"
                        )
                    reordered.append(source_idxs[match])
                    remaining.remove(match)
                aligned_idxs.append(reordered)
            idxs = aligned_idxs if "batch" in func.__name__ else aligned_idxs[0]
        if return_idx:
            return results, scores, idxs
        if return_score:
            return results, scores
        else:
            return results

    return wrapper


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.retrieval_method = self._config["retrieval_method"]
        self.topk = self._config["retrieval_topk"]

        self.index_path = self._config["index_path"]
        self.corpus_path = self._config["corpus_path"]

        self.save_cache = self._config["save_retrieval_cache"]
        self.use_cache = self._config["use_retrieval_cache"]
        self.cache_path = self._config["retrieval_cache_path"]

        self.use_reranker = self._config["use_reranker"]
        if self.use_reranker:
            self.reranker = get_reranker(self._config)
        else:
            self.reranker = None

        if self.save_cache:
            self.cache_save_path = os.path.join(
                self._config["save_dir"], "retrieval_cache.json"
            )
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        self.silent = (
            self._config["silent_retrieval"]
            if "silent_retrieval" in self._config
            else False
        )

    def update_additional_setting(self):
        raise NotImplementedError

    def _save_cache(self):
        self.cache = convert_numpy(self.cache)

        def custom_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4, default=custom_serializer)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        raise NotImplementedError

    def _batch_search(self, query, num, query_pool, return_score, return_idx):
        raise NotImplementedError

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class BaseTextRetriever(BaseRetriever):
    """Base text retriever."""

    def __init__(self, config):
        super().__init__(config)

    @cache_manager
    @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    @cache_manager
    @rerank_manager
    def batch_search(
        self, query, num=None, query_pool=None, return_score=False, return_idx=False
    ):
        return self._batch_search(
            query=query,
            num=num,
            query_pool=query_pool,
            return_score=return_score,
            return_idx=return_idx,
        )

    @rerank_manager
    def _batch_search_with_rerank(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _search_with_rerank(self, *args, **kwargs):
        return self._search(*args, **kwargs)


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)

        initialization_started = time.perf_counter()
        self.initialization_profile = {}

        self._cached_ivf_source = None
        self._cached_index_ivf = None
        self._cached_quantizer = None
        self._cached_centroids = None
        self._overlap_executor = None
        self._search_executor = None
        self._search_executor_workers = None
        self._retrieval_phase_totals = defaultdict(float)
        self._retrieval_phase_counts = defaultdict(int)
        self._retrieval_action_totals = defaultdict(int)
        self._retrieval_batch_count = 0
        self._guided_probe_observations = 0
        self._guided_probe_total = 0
        self._guided_probe_budget_total = 0
        self._guided_history_total = 0
        self._guided_history_novel_total = 0
        self._guided_history_added_total = 0
        self._guided_incomplete_fallback_total = 0
        self._guided_probe_histogram = defaultdict(int)
        self._guided_probe_budget_histogram = defaultdict(int)
        self._guided_max_codes = 0
        self._guided_probe_configuration = {}
        self._exact_duplicate_coalesced_total = 0
        self._exact_query_fast_path_batches = 0
        self._exact_query_mixed_fast_path_batches = 0
        self._exact_query_fast_path_requests = 0
        self._previous_batch_all_reuse = False
        self._reuse_first_batches = 0
        self._reuse_first_quantizer_skips = 0
        self._single_query_partitioned_searches = 0
        self._single_query_partition_total = 0
        self._single_query_partitioned_probe_total = 0

        phase_started = time.perf_counter()
        self.load_corpus(corpus)
        self.initialization_profile["corpus_load_s"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        self.load_index()
        self.initialization_profile["index_load_s"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        self.load_model()
        self.initialization_profile["encoder_load_s"] = (
            time.perf_counter() - phase_started
        )
        self.initialization_profile["total_s"] = (
            time.perf_counter() - initialization_started
        )
        print(
            "Retriever initialization profile: "
            + ", ".join(
                (
                    f"{name}={value}"
                    if isinstance(value, bool)
                    else f"{name}={value:.3f}s"
                )
                for name, value in self.initialization_profile.items()
            )
        )

    def _get_ivf_runtime(self):
        """Return cached IVF handles and long-lived retrieval executors."""
        try:
            max_workers = self._config["faiss_search_workers"]
        except Exception:
            max_workers = None
        max_workers = int(
            os.environ.get("MNEMORAG_FAISS_SEARCH_WORKERS")
            or max_workers
            or max(1, len(os.sched_getaffinity(0)) // faiss_omp_threads)
        )
        if max_workers < 1:
            raise ValueError("MNEMORAG_FAISS_SEARCH_WORKERS must be positive")

        if getattr(self, "_cached_ivf_source", None) is not self.index:
            self._cached_ivf_source = self.index
            self._cached_index_ivf = faiss.extract_index_ivf(self.index)
            self._cached_quantizer = self._cached_index_ivf.quantizer
            self._cached_centroids = np.ascontiguousarray(
                self._cached_quantizer.reconstruct_n(0, self._cached_index_ivf.nlist),
                dtype=np.float32,
            )

        if getattr(self, "_overlap_executor", None) is None:
            self._overlap_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="mnemorag-overlap",
            )

        if (
            getattr(self, "_search_executor", None) is None
            or getattr(self, "_search_executor_workers", None) != max_workers
        ):
            previous = getattr(self, "_search_executor", None)
            if previous is not None:
                previous.shutdown(wait=True)
            self._search_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="mnemorag-faiss",
            )
            self._search_executor_workers = max_workers

        return (
            self._cached_index_ivf,
            self._cached_quantizer,
            self._cached_centroids,
            self._overlap_executor,
            self._search_executor,
        )

    def _single_query_probe_partition_limit(self):
        """Return the MnemoRAG-only intra-query IVF partition limit."""
        try:
            enabled = bool(self._config["enable_single_query_probe_parallelism"])
        except Exception:
            enabled = False
        return 8 if enabled else 1

    @staticmethod
    def _search_preassigned(
        index_ivf, query, k, probes, centroid_distances, max_codes=0
    ):
        """Search exactly the supplied IVF lists with per-call parameters."""
        query = np.ascontiguousarray(query, dtype=np.float32)
        probes = np.ascontiguousarray(probes, dtype=np.int64)
        centroid_distances = np.ascontiguousarray(centroid_distances, dtype=np.float32)
        if probes.ndim == 1:
            probes = probes.reshape(1, -1)
        if centroid_distances.ndim == 1:
            centroid_distances = centroid_distances.reshape(1, -1)
        if probes.shape != centroid_distances.shape:
            raise ValueError("probes and centroid_distances must have equal shapes")
        if probes.shape[0] != query.shape[0] or probes.shape[1] < 1:
            raise ValueError("Invalid preassigned IVF probe shape")

        distances = np.empty((query.shape[0], k), dtype=np.float32)
        labels = np.empty((query.shape[0], k), dtype=np.int64)
        params = faiss.SearchParametersIVF()
        params.nprobe = probes.shape[1]
        params.max_codes = int(max_codes)
        index_ivf.search_preassigned_c(
            query.shape[0],
            faiss.swig_ptr(query),
            k,
            faiss.swig_ptr(probes),
            faiss.swig_ptr(centroid_distances),
            faiss.swig_ptr(distances),
            faiss.swig_ptr(labels),
            False,
            params,
        )
        return distances, labels

    @staticmethod
    def _merge_partitioned_topk(partition_results, k, metric_type):
        """Merge exact top-k candidates returned by disjoint IVF lists."""
        distances = np.concatenate(
            [np.asarray(result[0][0]) for result in partition_results]
        )
        labels = np.concatenate(
            [np.asarray(result[1][0]) for result in partition_results]
        )
        valid = labels >= 0
        distances = distances[valid]
        labels = labels[valid]
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            order = np.lexsort((labels, -distances))
            missing_distance = -np.inf
        elif metric_type == faiss.METRIC_L2:
            order = np.lexsort((labels, distances))
            missing_distance = np.inf
        else:
            raise NotImplementedError(
                "Partitioned IVF search supports inner-product and L2 indexes"
            )
        selected = order[:k]
        merged_distances = np.full((1, k), missing_distance, dtype=np.float32)
        merged_labels = np.full((1, k), -1, dtype=np.int64)
        merged_distances[0, : len(selected)] = distances[selected]
        merged_labels[0, : len(selected)] = labels[selected]
        return merged_distances, merged_labels

    def _search_preassigned_parallel(
        self,
        index_ivf,
        query,
        k,
        probes,
        centroid_distances,
        search_executor,
        *,
        max_partitions=8,
    ):
        """Scan one query's disjoint probe lists in parallel without waiting."""
        probes = np.asarray(probes, dtype=np.int64)
        centroid_distances = np.asarray(centroid_distances, dtype=np.float32)
        worker_limit = int(getattr(self, "_search_executor_workers", 1) or 1)
        partition_count = min(
            max(1, int(max_partitions)),
            worker_limit,
            len(probes),
        )
        if partition_count <= 1:
            return self._search_preassigned(
                index_ivf,
                query,
                k,
                probes,
                centroid_distances,
            )
        probe_partitions = np.array_split(probes, partition_count)
        distance_partitions = np.array_split(centroid_distances, partition_count)
        futures = [
            search_executor.submit(
                self._search_preassigned,
                index_ivf,
                query,
                k,
                probe_partition,
                distance_partition,
            )
            for probe_partition, distance_partition in zip(
                probe_partitions, distance_partitions
            )
        ]
        partition_results = [future.result() for future in futures]
        self._single_query_partitioned_searches = (
            int(getattr(self, "_single_query_partitioned_searches", 0)) + 1
        )
        self._single_query_partition_total = (
            int(getattr(self, "_single_query_partition_total", 0)) + partition_count
        )
        self._single_query_partitioned_probe_total = int(
            getattr(self, "_single_query_partitioned_probe_total", 0)
        ) + len(probes)
        return self._merge_partitioned_topk(partition_results, k, index_ivf.metric_type)

    @staticmethod
    def _centroid_distances(index_ivf, query, probes, centroids):
        selected = centroids[np.asarray(probes, dtype=np.int64)]
        if index_ivf.metric_type == faiss.METRIC_INNER_PRODUCT:
            return np.asarray(selected @ query[0], dtype=np.float32)
        if index_ivf.metric_type == faiss.METRIC_L2:
            delta = selected - query[0]
            return np.asarray(np.einsum("ij,ij->i", delta, delta), dtype=np.float32)
        raise NotImplementedError(
            "Preassigned IVF guidance supports inner-product and L2 indexes"
        )

    def close(self):
        """Release worker threads owned by the retriever."""
        for name in ("_overlap_executor", "_search_executor"):
            executor = getattr(self, name, None)
            if executor is not None:
                executor.shutdown(wait=True)
                setattr(self, name, None)

    def _record_retrieval_profile(
        self, phase_times, actions, guided_probe_metrics=None
    ):
        self.last_retrieval_phase_times = dict(phase_times)
        if not hasattr(self, "_retrieval_phase_totals"):
            self._retrieval_phase_totals = defaultdict(float)
            self._retrieval_phase_counts = defaultdict(int)
            self._retrieval_action_totals = defaultdict(int)
            self._retrieval_batch_count = 0
            self._guided_probe_observations = 0
            self._guided_probe_total = 0
            self._guided_probe_budget_total = 0
            self._guided_history_total = 0
            self._guided_history_novel_total = 0
            self._guided_history_added_total = 0
            self._guided_incomplete_fallback_total = 0
            self._guided_probe_histogram = defaultdict(int)
            self._guided_probe_budget_histogram = defaultdict(int)
            self._exact_duplicate_coalesced_total = 0
            self._exact_query_fast_path_batches = 0
            self._exact_query_mixed_fast_path_batches = 0
            self._exact_query_fast_path_requests = 0
            self._single_query_partitioned_searches = 0
            self._single_query_partition_total = 0
            self._single_query_partitioned_probe_total = 0
        for name, value in phase_times.items():
            self._retrieval_phase_totals[name] += float(value)
            self._retrieval_phase_counts[name] += 1
        for action in actions:
            self._retrieval_action_totals[action] += 1
        for metrics in guided_probe_metrics or ():
            if metrics is None:
                continue
            probe_count = int(metrics["nprobe"])
            probe_budget = int(metrics["probe_budget"])
            self._guided_probe_observations += 1
            self._guided_probe_total += probe_count
            self._guided_probe_budget_total += probe_budget
            self._guided_history_total += int(metrics["historical_clusters"])
            self._guided_history_novel_total += int(
                metrics["historical_novel_clusters"]
            )
            self._guided_history_added_total += int(
                metrics["historical_added_clusters"]
            )
            self._guided_incomplete_fallback_total += int(
                metrics.get("incomplete_result_fallback", False)
            )
            self._guided_probe_histogram[probe_count] += 1
            self._guided_probe_budget_histogram[probe_budget] += 1
        self._retrieval_batch_count += 1

    def retrieval_profile(self):
        totals = dict(getattr(self, "_retrieval_phase_totals", {}))
        counts = dict(getattr(self, "_retrieval_phase_counts", {}))
        guided_count = int(getattr(self, "_guided_probe_observations", 0))
        return {
            "batches": int(getattr(self, "_retrieval_batch_count", 0)),
            "phase_total_s": totals,
            "phase_mean_s": {
                name: totals[name] / counts[name]
                for name in totals
                if counts.get(name, 0)
            },
            "action_counts": dict(getattr(self, "_retrieval_action_totals", {})),
            "exact_duplicate_requests_coalesced": int(
                getattr(self, "_exact_duplicate_coalesced_total", 0)
            ),
            "exact_query_fast_path": {
                "batches": int(getattr(self, "_exact_query_fast_path_batches", 0)),
                "mixed_batches": int(
                    getattr(
                        self,
                        "_exact_query_mixed_fast_path_batches",
                        0,
                    )
                ),
                "requests": int(getattr(self, "_exact_query_fast_path_requests", 0)),
            },
            "adaptive_reuse_first": {
                "batches": int(getattr(self, "_reuse_first_batches", 0)),
                "quantizer_skips": int(
                    getattr(self, "_reuse_first_quantizer_skips", 0)
                ),
            },
            "single_query_probe_parallelism": {
                "enabled": (self._single_query_probe_partition_limit() > 1),
                "searches": int(getattr(self, "_single_query_partitioned_searches", 0)),
                "partitions_mean": (
                    self._single_query_partition_total
                    / self._single_query_partitioned_searches
                    if getattr(self, "_single_query_partitioned_searches", 0)
                    else 0.0
                ),
                "probes_mean": (
                    self._single_query_partitioned_probe_total
                    / self._single_query_partitioned_searches
                    if getattr(self, "_single_query_partitioned_searches", 0)
                    else 0.0
                ),
                "maximum_partitions": (self._single_query_probe_partition_limit()),
            },
            "guided_probe_stats": {
                "observations": guided_count,
                "max_codes": int(getattr(self, "_guided_max_codes", 0)),
                "configuration": dict(getattr(self, "_guided_probe_configuration", {})),
                "nprobe_mean": (
                    self._guided_probe_total / guided_count if guided_count else None
                ),
                "probe_budget_mean": (
                    self._guided_probe_budget_total / guided_count
                    if guided_count
                    else None
                ),
                "historical_clusters_mean": (
                    self._guided_history_total / guided_count if guided_count else None
                ),
                "historical_novel_clusters_mean": (
                    self._guided_history_novel_total / guided_count
                    if guided_count
                    else None
                ),
                "historical_added_clusters_mean": (
                    self._guided_history_added_total / guided_count
                    if guided_count
                    else None
                ),
                "incomplete_result_fallbacks": int(
                    getattr(self, "_guided_incomplete_fallback_total", 0)
                ),
                "nprobe_histogram": {
                    str(key): value
                    for key, value in sorted(
                        getattr(self, "_guided_probe_histogram", {}).items()
                    )
                },
                "probe_budget_histogram": {
                    str(key): value
                    for key, value in sorted(
                        getattr(self, "_guided_probe_budget_histogram", {}).items()
                    )
                },
            },
        }

    def reset_retrieval_profile(self):
        """Reset online profiling counters without rebuilding the retriever."""
        self._retrieval_phase_totals = defaultdict(float)
        self._retrieval_phase_counts = defaultdict(int)
        self._retrieval_action_totals = defaultdict(int)
        self._retrieval_batch_count = 0
        self._guided_probe_observations = 0
        self._guided_probe_total = 0
        self._guided_probe_budget_total = 0
        self._guided_history_total = 0
        self._guided_history_novel_total = 0
        self._guided_history_added_total = 0
        self._guided_incomplete_fallback_total = 0
        self._guided_probe_histogram = defaultdict(int)
        self._guided_probe_budget_histogram = defaultdict(int)
        self._reuse_first_batches = 0
        self._reuse_first_quantizer_skips = 0
        self._exact_query_fast_path_batches = 0
        self._exact_query_mixed_fast_path_batches = 0
        self._exact_query_fast_path_requests = 0
        self._exact_duplicate_coalesced_total = 0
        self._single_query_partitioned_searches = 0
        self._single_query_partition_total = 0
        self._single_query_partitioned_probe_total = 0

    def load_corpus(self, corpus):
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus

    def load_index(self):
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        try:
            configured_mmap = bool(self._config["faiss_mmap_index"])
        except Exception:
            configured_mmap = False
        use_mmap = (
            os.environ.get(
                "MNEMORAG_FAISS_MMAP_INDEX",
                "1" if configured_mmap else "0",
            )
            == "1"
        )
        if use_mmap:
            flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            self.index = faiss.read_index(self.index_path, flags)
        else:
            self.index = faiss.read_index(self.index_path)
        self.initialization_profile["index_mmap"] = use_mmap
        print("index type: ", type(self.index))
        print(f"faiss mmap index: {use_mmap}")
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = self._config["ivf_nprobe"]
            mapping_started = time.perf_counter()
            self.docid_to_cluster = self._build_docid_to_cluster(self.index)
            mapping_elapsed = time.perf_counter() - mapping_started
            if isinstance(self.docid_to_cluster, np.ndarray):
                mapping_bytes = self.docid_to_cluster.nbytes
                mapping_kind = "dense-int32"
            else:
                mapping_bytes = None
                mapping_kind = "sparse-dict"
            print(
                "docid_to_cluster mapping: "
                f"kind={mapping_kind}, entries={len(self.docid_to_cluster)}, "
                f"bytes={mapping_bytes}, time={mapping_elapsed:.3f}s"
            )
        else:
            self.docid_to_cluster = None
            print("index is not IVF, skip cluster mapping")

        if self.use_faiss_gpu:
            print("Using faiss GPU for retrieval")
            index_cpu_to_all_gpus_time_st = time.time()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            index_cpu_to_all_gpus_time_ed = time.time()
            print(
                f"index_cpu_to_all_gpus_time: {index_cpu_to_all_gpus_time_ed - index_cpu_to_all_gpus_time_st} seconds"
            )

    @staticmethod
    def _build_docid_to_cluster(index):
        """Build a compact mapping for dense ids and preserve sparse-id support."""
        invlists = index.invlists
        dense_mapping = np.full(int(index.ntotal), -1, dtype=np.int32)
        dense_ids = True

        for list_id in range(index.nlist):
            list_size = invlists.list_size(list_id)
            if list_size == 0:
                continue
            ids = np.asarray(faiss.rev_swig_ptr(invlists.get_ids(list_id), list_size))
            if np.any(ids < 0) or np.any(ids >= index.ntotal):
                dense_ids = False
                break
            dense_mapping[ids] = list_id

        if dense_ids:
            return dense_mapping

        sparse_mapping = {}
        for list_id in range(index.nlist):
            list_size = invlists.list_size(list_id)
            if list_size == 0:
                continue
            ids = faiss.rev_swig_ptr(invlists.get_ids(list_id), list_size)
            sparse_mapping.update((int(doc_id), list_id) for doc_id in ids)
        return sparse_mapping

    def _clusters_for_doc_ids(self, doc_ids):
        if self.docid_to_cluster is None:
            return None
        if isinstance(self.docid_to_cluster, np.ndarray):
            ids = np.asarray(doc_ids, dtype=np.int64)
            valid = (ids >= 0) & (ids < len(self.docid_to_cluster))
            clusters = np.full(ids.shape, -1, dtype=np.int64)
            clusters[valid] = self.docid_to_cluster[ids[valid]]
            return [int(cluster) for cluster in clusters if cluster >= 0]
        return [
            int(self.docid_to_cluster[doc_id])
            for doc_id in doc_ids
            if doc_id in self.docid_to_cluster
        ]

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = self._config["instruction"]

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def load_model(self):
        if self.use_st:
            raise ValueError(
                "The paper runtime uses the native E5 encoder, not "
                "SentenceTransformers."
            )
        self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=self.retrieval_model_path,
            pooling_method=self.pooling_method,
            max_length=self.query_max_length,
            use_fp16=self.use_fp16,
            instruction=self.instruction,
            silent=self.silent,
        )
        print(
            "Encoder model local in :",
            next(self.encoder.model.parameters()).device,
        )

    def _check_pooling_method(self, model_path, pooling_method):
        detect_pooling_method = None
        try:
            pooling_path = os.path.join(model_path, "1_Pooling/config.json")
            with open(pooling_path, encoding="utf-8") as stream:
                pooling_config = json.load(stream)
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(
                            f"Pooling method {detect_pooling_method} is not implemented.",
                            UserWarning,
                        )
                        detect_pooling_method = "mean"
                    break
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

        if (
            detect_pooling_method is not None
            and detect_pooling_method != pooling_method
        ):
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(
        self,
        query: str,
        num: int = None,
        query_pool: QueryPool = None,
        return_score=False,
        return_idx=False,
    ):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        results = load_docs(self.corpus, idxs)
        if return_idx:
            return results, scores, idxs
        if return_score:
            return results, scores
        else:
            return results

    def __batch_search(
        self,
        query,
        num=None,
        query_pool: QueryPool = None,
        return_score=False,
        return_idx=False,
    ):
        retrieval_start = time.time()
        if isinstance(query, str):
            query = [query]

        if num is None:
            num = self.topk

        batch_size = self.batch_size
        results = []
        scores = []
        encode_time_st = time.time()
        emb = self.encoder.encode(query, batch_size=batch_size, is_query=True)
        encode_time_ed = time.time()

        print(
            f"Encoding time for {len(query)} samples: {encode_time_ed - encode_time_st:.4f} seconds"
        )

        search_time_st = time.time()
        scores, idxs = self.index.search(emb, k=num)
        search_time_ed = time.time()

        print(
            f"Faiss search time for {len(query)} samples: {search_time_ed - search_time_st:.4f} seconds"
        )

        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = [idx for sublist in idxs for idx in sublist]

        load_docs_time_st = time.time()
        results = load_docs(self.corpus, flat_idxs)
        load_docs_time_ed = time.time()

        print(
            f"load_docs time: {load_docs_time_ed - load_docs_time_st:.4f} seconds"
        )

        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]
        retrieval_end = time.time()
        print(
            f"Retrieval time for {len(query)} samples: {retrieval_end - retrieval_start:.4f}s"
        )
        if return_idx:
            return results, scores, idxs

        if return_score:
            return results, scores

        return results

    # Batched IVF retrieval with one OpenMP thread per worker.
    def _batch_search(
        self,
        query: List[str],
        num: int = None,
        query_pool: QueryPool = None,
        return_score=False,
        return_idx=False,
    ):
        retrieval_start = time.perf_counter()
        phase_times = {}
        self.last_request_latency_ms = []
        self.last_request_search_service_ms = []
        self.last_request_actions = []
        self.last_request_latency_definition = (
            "request admission to query-specific candidate completion; "
            "includes shared batch preprocessing and executor queueing, but "
            "does not wait for unrelated queries to finish"
        )

        def publish_request_timings(completion_s, service_s, action_names):
            if not (len(completion_s) == len(service_s) == len(action_names)):
                raise RuntimeError("Per-request retrieval timing shape mismatch")
            if any(value is None for value in completion_s):
                raise RuntimeError("Missing per-request retrieval completion time")
            self.last_request_latency_ms = [
                1000.0 * float(value) for value in completion_s
            ]
            self.last_request_search_service_ms = [
                1000.0 * float(value) for value in service_s
            ]
            self.last_request_actions = list(action_names)

        debug_timing = os.environ.get("MNEMORAG_RETRIEVAL_DEBUG_TIMING", "0") == "1"

        def debug(message):
            if debug_timing:
                print(message)

        # basic setup
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk

        unique_queries = []
        query_positions = {}
        inverse_positions = []
        for query_text in query:
            if query_text not in query_positions:
                position = len(unique_queries)
                query_positions[query_text] = position
                unique_queries.append(query_text)
            else:
                position = query_positions[query_text]
            inverse_positions.append(position)
        duplicate_count = len(query) - len(unique_queries)
        if duplicate_count:
            unique_results, unique_scores, unique_idxs = self._batch_search(
                unique_queries,
                num=num,
                query_pool=query_pool,
                return_score=True,
                return_idx=True,
            )
            unique_latencies = list(self.last_request_latency_ms)
            unique_services = list(self.last_request_search_service_ms)
            unique_actions = list(self.last_request_actions)
            publish_request_timings(
                [unique_latencies[pos] / 1000.0 for pos in inverse_positions],
                [unique_services[pos] / 1000.0 for pos in inverse_positions],
                [unique_actions[pos] for pos in inverse_positions],
            )
            self._exact_duplicate_coalesced_total += duplicate_count
            self.last_exact_duplicate_coalesced = duplicate_count
            results = [list(unique_results[pos]) for pos in inverse_positions]
            scores = [list(unique_scores[pos]) for pos in inverse_positions]
            idxs = [list(unique_idxs[pos]) for pos in inverse_positions]
            if return_idx:
                return results, scores, idxs
            if return_score:
                return results, scores
            return results
        self.last_exact_duplicate_coalesced = 0

        B = len(query)
        batch_size = self.batch_size

        exact_slots = None
        if query_pool is not None and getattr(query_pool, "enable_exact_lookup", False):
            exact_lookup_started = time.perf_counter()
            exact_slots = query_pool.lookup_exact_batch(query)
            phase_times["exact_lookup_s"] = time.perf_counter() - exact_lookup_started
        exact_hits = []
        for position, slot in enumerate(exact_slots or ()):
            if slot is None:
                continue
            state = query_pool.query_states[int(slot)]
            if (
                state.topk_doc_ids is not None
                and state.topk_scores is not None
                and len(state.topk_doc_ids) >= num
                and len(state.topk_scores) >= num
            ):
                # Copy the compact arrays before processing misses because a
                # capacity-limited insertion may replace their Query Base slot.
                exact_hits.append(
                    (
                        position,
                        int(slot),
                        list(state.topk_doc_ids[:num]),
                        list(state.topk_scores[:num]),
                    )
                )

        if exact_hits:
            query_pool.touch([hit[1] for hit in exact_hits])
            hit_count = len(exact_hits)

            if hit_count == B:
                idxs = [hit[2] for hit in exact_hits]
                scores = [hit[3] for hit in exact_hits]
                flat_idxs = [doc_id for row in idxs for doc_id in row]
                load_started = time.perf_counter()
                docs = load_docs(self.corpus, flat_idxs)
                phase_times["load_docs_s"] = time.perf_counter() - load_started
                results = [docs[index * num : (index + 1) * num] for index in range(B)]
                actions = ["exact"] * B
                self.last_similarity_action_counts = {
                    "exact": B,
                    "reuse": 0,
                    "guided": 0,
                    "full": 0,
                }
                self._previous_batch_all_reuse = True
                self.last_retrieval_elapsed = time.perf_counter() - retrieval_start
                publish_request_timings(
                    [self.last_retrieval_elapsed] * B,
                    [0.0] * B,
                    ["exact"] * B,
                )
                phase_times["total_s"] = self.last_retrieval_elapsed
                self._record_retrieval_profile(phase_times, actions)
                self._exact_query_fast_path_batches = (
                    getattr(self, "_exact_query_fast_path_batches", 0) + 1
                )
                self._exact_query_fast_path_requests = (
                    getattr(self, "_exact_query_fast_path_requests", 0) + hit_count
                )
                if return_idx:
                    return results, scores, idxs
                if return_score:
                    return results, scores
                return results

            hit_positions = {hit[0] for hit in exact_hits}
            miss_positions = [
                position for position in range(B) if position not in hit_positions
            ]
            exact_ready_s = time.perf_counter() - retrieval_start
            miss_start_offset_s = time.perf_counter() - retrieval_start
            miss_results, miss_scores, miss_idxs = self._batch_search(
                [query[position] for position in miss_positions],
                num=num,
                query_pool=query_pool,
                return_score=True,
                return_idx=True,
            )
            miss_latencies_ms = list(self.last_request_latency_ms)
            miss_services_ms = list(self.last_request_search_service_ms)
            miss_actions = list(self.last_request_actions)
            self._exact_query_fast_path_batches = (
                getattr(self, "_exact_query_fast_path_batches", 0) + 1
            )
            self._exact_query_mixed_fast_path_batches = (
                getattr(self, "_exact_query_mixed_fast_path_batches", 0) + 1
            )
            self._exact_query_fast_path_requests = (
                getattr(self, "_exact_query_fast_path_requests", 0) + hit_count
            )
            inner_elapsed = self.last_retrieval_elapsed
            results = [None] * B
            scores = [None] * B
            idxs = [None] * B
            for offset, position in enumerate(miss_positions):
                results[position] = miss_results[offset]
                scores[position] = miss_scores[offset]
                idxs[position] = miss_idxs[offset]

            exact_load_started = time.perf_counter()
            exact_doc_ids = [doc_id for hit in exact_hits for doc_id in hit[2]]
            exact_docs = load_docs(self.corpus, exact_doc_ids)
            exact_load_s = time.perf_counter() - exact_load_started
            for offset, (position, _slot, hit_idxs, hit_scores) in enumerate(
                exact_hits
            ):
                results[position] = exact_docs[offset * num : (offset + 1) * num]
                scores[position] = hit_scores
                idxs[position] = hit_idxs

            inner_counts = dict(getattr(self, "last_similarity_action_counts", {}))
            self.last_similarity_action_counts = {
                "exact": hit_count,
                "reuse": int(inner_counts.get("reuse", 0)),
                "guided": int(inner_counts.get("guided", 0)),
                "full": int(inner_counts.get("full", 0)),
            }
            self._retrieval_action_totals["exact"] += hit_count
            self.last_retrieval_elapsed = time.perf_counter() - retrieval_start
            completion_s = [exact_ready_s] * B
            service_s = [0.0] * B
            request_actions = ["exact"] * B
            for offset, position in enumerate(miss_positions):
                completion_s[position] = (
                    miss_start_offset_s + miss_latencies_ms[offset] / 1000.0
                )
                service_s[position] = miss_services_ms[offset] / 1000.0
                request_actions[position] = miss_actions[offset]
            publish_request_timings(completion_s, service_s, request_actions)

            # The recursive miss path records the logical batch. Fold the
            # exact-hit work into that profile without double-counting it.
            outer_lookup_s = phase_times.get("exact_lookup_s", 0.0)
            outer_delta_s = max(0.0, self.last_retrieval_elapsed - inner_elapsed)
            self._retrieval_phase_totals["exact_lookup_s"] += outer_lookup_s
            self._retrieval_phase_totals["load_docs_s"] += exact_load_s
            self._retrieval_phase_totals["total_s"] += outer_delta_s
            last_phases = dict(getattr(self, "last_retrieval_phase_times", {}))
            last_phases["exact_lookup_s"] = (
                last_phases.get("exact_lookup_s", 0.0) + outer_lookup_s
            )
            last_phases["load_docs_s"] = (
                last_phases.get("load_docs_s", 0.0) + exact_load_s
            )
            last_phases["total_s"] = self.last_retrieval_elapsed
            self.last_retrieval_phase_times = last_phases
            if return_idx:
                return results, scores, idxs
            if return_score:
                return results, scores
            return results

        # thresholds & params
        T_REUSE, T_GUIDED = _resolve_similarity_thresholds(self._config)

        try:
            enable_similarity_aware = bool(self._config["enable_similarity_aware"])
        except Exception:
            enable_similarity_aware = True
        debug(f"enable_similarity_aware={enable_similarity_aware}")

        SIM_LOW = T_GUIDED
        SIM_HIGH = T_REUSE

        def optional_config(name, default=None):
            try:
                value = self._config[name]
            except (KeyError, TypeError):
                return default
            return default if value is None else value

        base_nprobe = int(self._config["ivf_nprobe"])
        configured_nprobe_max = optional_config("similarity_nprobe_max")
        NPROBE_MAX = (
            max(1, base_nprobe // 2)
            if configured_nprobe_max is None
            else int(configured_nprobe_max)
        )
        configured_nprobe_min = optional_config("similarity_nprobe_min")
        NPROBE_MIN = (
            min(16, NPROBE_MAX)
            if configured_nprobe_min is None
            else int(configured_nprobe_min)
        )
        if not 1 <= NPROBE_MIN <= NPROBE_MAX <= base_nprobe:
            raise ValueError(
                "Adaptive nprobe bounds must satisfy 1 <= min <= max <= ivf_nprobe"
            )
        GAMMA = float(optional_config("similarity_nprobe_gamma", 2.0))
        NPROBE_CURVE = str(
            optional_config("similarity_nprobe_curve", "power_decay")
        ).lower()
        if NPROBE_CURVE not in {"power_decay", "late_decay"}:
            raise ValueError(
                "similarity_nprobe_curve must be power_decay or late_decay"
            )
        NORMALIZATION = str(
            optional_config("similarity_nprobe_normalization", "cumulative")
        ).lower()
        if NORMALIZATION not in {"threshold", "cumulative"}:
            raise ValueError(
                "similarity_nprobe_normalization must be threshold or cumulative"
            )
        NPROBE_ROUNDING = str(
            optional_config("similarity_nprobe_rounding", "ceil")
        ).lower()
        self._guided_probe_configuration = {
            "normalization": NORMALIZATION,
            "rounding": NPROBE_ROUNDING,
            "gamma": GAMMA,
            "curve": NPROBE_CURVE,
            "minimum": NPROBE_MIN,
            "maximum_query_budget": NPROBE_MAX,
            "historical_clusters": (
                "disabled_when_full_query_budget"
                if NPROBE_MIN == base_nprobe == NPROBE_MAX
                else "additive_outside_query_budget"
            ),
        }
        try:
            configured_max_codes = int(self._config["faiss_max_codes"] or 0)
        except Exception:
            configured_max_codes = 0
        guided_max_codes = int(
            os.environ.get("MNEMORAG_FAISS_MAX_CODES") or configured_max_codes
        )
        if guided_max_codes < 0:
            raise ValueError("MNEMORAG_FAISS_MAX_CODES must be nonnegative")
        self._guided_max_codes = guided_max_codes

        # Encode (GPU)
        encode_time_st = time.perf_counter()
        encoded = self.encoder.encode(
            query,
            batch_size=batch_size,
            is_query=True,
            return_torch=True,
        )
        if torch.is_tensor(encoded):
            similarity_emb = encoded
            emb = encoded.detach().cpu().numpy().astype(np.float32, order="C")
        else:
            # Compatibility with external/fake encoders that only expose NumPy.
            emb = np.asarray(encoded, dtype=np.float32, order="C")
            similarity_emb = emb
        encode_time_ed = time.perf_counter()
        phase_times["encode_s"] = encode_time_ed - encode_time_st
        debug(
            f"Encoding time for {B} samples: {phase_times['encode_s']:.4f}s"
        )

        # IVF & quantizer
        (
            index_ivf,
            quantizer,
            coarse_centroids,
            overlap_executor,
            search_executor,
        ) = self._get_ivf_runtime()

        # Historical-query matching and coarse search run concurrently.
        def run_similarity():
            run_similarity_st_time = time.perf_counter()
            sim_scores, sim_idxs = None, None
            if query_pool is not None and query_pool.current_size() > 0:
                sim_scores, sim_idxs = query_pool.search_similar_batch(
                    similarity_emb, topk=1
                )
            run_similarity_ed_time = time.perf_counter()
            phase_times["similarity_s"] = (
                run_similarity_ed_time - run_similarity_st_time
            )
            debug(
                f"Similarity search time for {B} samples: "
                f"{phase_times['similarity_s']:.4f}s"
            )
            return sim_scores, sim_idxs

        def run_quantizer(faiss_indices):
            run_quantizer_st_time = time.perf_counter()
            if not faiss_indices:
                return None, None

            X = emb[faiss_indices]
            distances, results = quantizer.search(X, NPROBE_MAX)
            run_quantizer_ed_time = time.perf_counter()
            phase_times["quantizer_s"] = run_quantizer_ed_time - run_quantizer_st_time
            debug(
                f"Quantizer search time for {len(faiss_indices)} samples: "
                f"{phase_times['quantizer_s']:.4f}s"
            )
            return distances, results

        # init outputs
        scores = [None] * B
        idxs = [None] * B

        # Submit independent GPU and CPU preparation tasks.
        parallel_st = time.perf_counter()

        if (
            enable_similarity_aware
            and query_pool is not None
            and query_pool.current_size() > 0
        ):
            adaptive_reuse_first = os.environ.get(
                "MNEMORAG_ADAPTIVE_REUSE_FIRST_QUANTIZER", "0"
            ) == "1" and bool(getattr(self, "_previous_batch_all_reuse", False))
            if adaptive_reuse_first:
                self._reuse_first_batches = getattr(self, "_reuse_first_batches", 0) + 1
                sim_scores, sim_idxs = run_similarity()
                if bool(torch.all(sim_scores[:, 0] >= T_REUSE)):
                    self_D_all, self_I_all = None, None
                    phase_times["quantizer_s"] = 0.0
                    self._reuse_first_quantizer_skips = (
                        getattr(self, "_reuse_first_quantizer_skips", 0) + 1
                    )
                else:
                    self_D_all, self_I_all = run_quantizer(list(range(B)))
            else:
                f_sim = overlap_executor.submit(run_similarity)
                f_quant = overlap_executor.submit(run_quantizer, list(range(B)))
                sim_scores, sim_idxs = f_sim.result()
                self_D_all, self_I_all = f_quant.result()

            parallel_ed = time.perf_counter()
            phase_times["similarity_quantizer_wall_s"] = parallel_ed - parallel_st
            debug(
                f"GPU SIM + QUANT parallel time for {B} samples: "
                f"{phase_times['similarity_quantizer_wall_s']:.4f}s"
            )
        else:
            sim_scores, sim_idxs = None, None
            self_D_all, self_I_all = None, None
            phase_times["similarity_s"] = 0.0
            phase_times["quantizer_s"] = 0.0
            phase_times["similarity_quantizer_wall_s"] = 0.0
            debug(
                "Similarity-Aware Retrieval bypassed: using full IVF search"
            )

        if sim_scores is not None:
            max_sim = sim_scores[:, 0]
            query_pool.min_similarity = min(
                query_pool.min_similarity, max_sim.min().item()
            )
            query_pool.max_similarity = max(
                query_pool.max_similarity, max_sim.max().item()
            )
            if (
                NORMALIZATION == "cumulative"
                and query_pool.max_similarity > query_pool.min_similarity
            ):
                SIM_LOW = float(query_pool.min_similarity)
                SIM_HIGH = float(query_pool.max_similarity)
            debug(
                f"Current similarity normalization range: "
                f"[{SIM_LOW:.4f}, {SIM_HIGH:.4f}]"
            )
        # Action decision
        actions = ["full"] * B
        request_completion_s = [None] * B
        request_search_service_s = [0.0] * B
        guided_probe_metrics = [None] * B
        if sim_scores is not None:
            for i in range(B):
                s = sim_scores[i, 0].item()
                if s >= T_REUSE:
                    actions[i] = "reuse"
                elif s >= T_GUIDED:
                    actions[i] = "guided"
        if query_pool is not None and sim_idxs is not None:
            for i, action_name in enumerate(actions):
                if action_name != "reuse":
                    continue
                hist_state = query_pool.query_states[sim_idxs[i, 0].item()]
                if (
                    hist_state.topk_doc_ids is None
                    or hist_state.topk_scores is None
                    or len(hist_state.topk_doc_ids) < num
                    or len(hist_state.topk_scores) < num
                ):
                    # A smaller historical top-k cannot satisfy a deeper
                    # request. Full IVF preserves the requested result depth
                    # without relying on an incomplete cached payload.
                    actions[i] = "full"
        decision_end = time.perf_counter()
        phase_times["decision_s"] = (
            decision_end
            - parallel_st
            - phase_times.get("similarity_quantizer_wall_s", 0.0)
        )
        self.last_similarity_action_counts = {
            "reuse": actions.count("reuse"),
            "guided": actions.count("guided"),
            "full": actions.count("full"),
        }
        self._previous_batch_all_reuse = bool(actions) and all(
            action == "reuse" for action in actions
        )
        debug(f"thresholds: T_REUSE={T_REUSE:.2f}, T_GUIDED={T_GUIDED:.2f}")
        debug(
            f"decision: "
            f"reuse={actions.count('reuse')}, "
            f"guided={actions.count('guided')}, "
            f"full={actions.count('full')}"
        )

        # Refresh only historical entries that contributed to this retrieval.
        if query_pool is not None and sim_idxs is not None:
            used_slots = [
                sim_idxs[i, 0].item()
                for i, action in enumerate(actions)
                if action in ("reuse", "guided")
            ]
            query_pool.touch(used_slots)

        # reuse
        if query_pool is not None and sim_idxs is not None:
            for i in range(B):
                if actions[i] == "reuse":
                    reuse_started = time.perf_counter()
                    hist_state = query_pool.query_states[sim_idxs[i, 0].item()]
                    idxs[i] = list(hist_state.topk_doc_ids[:num])
                    scores[i] = list(hist_state.topk_scores[:num])
                    reuse_finished = time.perf_counter()
                    request_completion_s[i] = reuse_finished - retrieval_start
                    request_search_service_s[i] = reuse_finished - reuse_started

        # faiss needed indices
        faiss_indices = [i for i in range(B) if actions[i] != "reuse"]
        if not faiss_indices:
            # build results from reused idxs
            flat_idxs = [d for sub in idxs if sub for d in sub]
            load_st = time.perf_counter()
            docs = load_docs(self.corpus, flat_idxs)
            phase_times["load_docs_s"] = time.perf_counter() - load_st

            results = [docs[i * num : (i + 1) * num] for i in range(B)]

            self.last_retrieval_elapsed = time.perf_counter() - retrieval_start
            publish_request_timings(
                request_completion_s, request_search_service_s, actions
            )
            phase_times["total_s"] = self.last_retrieval_elapsed
            self._record_retrieval_profile(phase_times, actions, guided_probe_metrics)
            if return_idx:
                return results, scores, idxs
            if return_score:
                return results, scores
            return results

        # IVF search (per-query parallel)
        def search_one(local_i, qi, single_query_partitions=1):
            request_search_started = time.perf_counter()
            xq = emb[qi : qi + 1]
            if actions[qi] == "full":
                if single_query_partitions > 1 and hasattr(
                    index_ivf, "search_preassigned_c"
                ):
                    full_distances, full_probes = quantizer.search(xq, base_nprobe)
                    distances, indices = self._search_preassigned_parallel(
                        index_ivf,
                        xq,
                        num,
                        full_probes[0],
                        full_distances[0],
                        search_executor,
                        max_partitions=single_query_partitions,
                    )
                else:
                    distances, indices = index_ivf.search(xq, k=num)
                if np.any(indices[0] < 0):
                    raise RuntimeError(
                        "Faiss returned fewer valid documents than requested "
                        f"for a full IVF search (topk={num}, ntotal={index_ivf.ntotal})"
                    )
                request_search_finished = time.perf_counter()
                return (
                    qi,
                    distances[0].tolist(),
                    indices[0].tolist(),
                    request_search_started,
                    request_search_finished,
                )

            sim = sim_scores[qi, 0].item()
            normalization_low = SIM_LOW
            normalization_high = SIM_HIGH
            if normalization_high <= normalization_low:
                normalization_low = T_GUIDED
                normalization_high = T_REUSE
            nprobe = adaptive_probe_budget(
                sim,
                normalization_low,
                normalization_high,
                NPROBE_MIN,
                NPROBE_MAX,
                gamma=GAMMA,
                rounding=NPROBE_ROUNDING,
                curve=NPROBE_CURVE,
            )
            hist_state = query_pool.query_states[sim_idxs[qi, 0].item()]
            hist_clusters = list(hist_state.topk_cluster_ids or ())
            # self_I_all is encoded in original batch order. ``local_i`` is
            # only the position among non-reused queries and diverges from
            # ``qi`` whenever an earlier query takes the reuse path.
            self_clusters = self_I_all[qi].tolist()
            self_cluster_distances = self_D_all[qi].tolist()

            # Algorithm 1 defines P = C_pruned union H(v). Preserve the full
            # query-specific budget and append only novel historical lists;
            # counting history inside the budget can evict the query's nearest
            # lists and silently reduce retrieval quality.
            query_probes = _merge_guided_probe_lists(
                self_clusters[:nprobe],
                (),
                allow_historical_expansion=False,
            )
            probe_budget = len(query_probes)
            union_probes = _merge_guided_probe_lists(
                query_probes,
                hist_clusters,
                allow_historical_expansion=not (
                    NPROBE_MIN == base_nprobe == NPROBE_MAX
                ),
            )

            try:
                configured_preassigned = bool(
                    self._config["enable_preassigned_guidance"]
                )
            except Exception:
                configured_preassigned = True
            use_preassigned = (
                os.environ.get(
                    "MNEMORAG_FAISS_PREASSIGNED_GUIDANCE",
                    "1" if configured_preassigned else "0",
                )
                == "1"
            )
            probes = union_probes if use_preassigned else query_probes
            historical = set(hist_clusters)
            current = set(self_clusters)
            selected = set(probes)
            guided_probe_metrics[qi] = {
                "nprobe": len(probes),
                "probe_budget": probe_budget,
                "historical_clusters": len(historical & selected),
                "historical_novel_clusters": len((historical & selected) - current),
                "historical_added_clusters": len(probes) - probe_budget,
                "incomplete_result_fallback": False,
            }

            if use_preassigned:
                query_distance_by_cluster = dict(
                    zip(self_clusters, self_cluster_distances)
                )
                historical_only = [
                    cluster
                    for cluster in probes
                    if cluster not in query_distance_by_cluster
                ]
                if historical_only:
                    historical_distances = self._centroid_distances(
                        index_ivf,
                        xq,
                        historical_only,
                        coarse_centroids,
                    )
                    query_distance_by_cluster.update(
                        zip(historical_only, historical_distances.tolist())
                    )
                coarse_distances = np.asarray(
                    [query_distance_by_cluster[cluster] for cluster in probes],
                    dtype=np.float32,
                )
                search_args = (
                    index_ivf,
                    xq,
                    num,
                    np.asarray(probes, dtype=np.int64),
                    coarse_distances,
                )
                if guided_max_codes:
                    distances, indices = self._search_preassigned(
                        *search_args, max_codes=guided_max_codes
                    )
                elif single_query_partitions > 1 and hasattr(
                    index_ivf, "search_preassigned_c"
                ):
                    distances, indices = self._search_preassigned_parallel(
                        index_ivf,
                        xq,
                        num,
                        np.asarray(probes, dtype=np.int64),
                        coarse_distances,
                        search_executor,
                        max_partitions=single_query_partitions,
                    )
                else:
                    distances, indices = self._search_preassigned(*search_args)
            else:
                params = faiss.SearchParametersIVF()
                params.nprobe = probe_budget
                params.max_codes = guided_max_codes
                distances, indices = index_ivf.search(xq, k=num, params=params)
            if np.any(indices[0] < 0):
                # Restricted IVF lists may contain fewer than top-k vectors.
                # Fall back only for the affected query so invalid Faiss labels
                # can never be materialized as Python's negative indices.
                distances, indices = index_ivf.search(xq, k=num)
                guided_probe_metrics[qi]["incomplete_result_fallback"] = True
                if np.any(indices[0] < 0):
                    raise RuntimeError(
                        "Faiss returned fewer valid documents than requested "
                        f"after full-search fallback (topk={num}, "
                        f"ntotal={index_ivf.ntotal})"
                    )
            request_search_finished = time.perf_counter()
            return (
                qi,
                distances[0].tolist(),
                indices[0].tolist(),
                request_search_started,
                request_search_finished,
            )

        cpu_time_st = time.perf_counter()
        batch_full_search = os.environ.get(
            "MNEMORAG_FAISS_BATCH_FULL", "0"
        ) == "1" and all(actions[qi] == "full" for qi in faiss_indices)
        if batch_full_search:
            batch_search_started = time.perf_counter()
            batch_emb = emb[faiss_indices]
            batch_scores, batch_idxs = index_ivf.search(batch_emb, k=num)
            batch_search_finished = time.perf_counter()
            for local_i, qi in enumerate(faiss_indices):
                scores[qi] = batch_scores[local_i].tolist()
                idxs[qi] = batch_idxs[local_i].tolist()
                request_completion_s[qi] = batch_search_finished - retrieval_start
                request_search_service_s[qi] = (
                    batch_search_finished - batch_search_started
                )
        elif len(faiss_indices) == 1:
            qi, score, idx, request_started, request_finished = search_one(
                0,
                faiss_indices[0],
                self._single_query_probe_partition_limit(),
            )
            scores[qi] = score
            idxs[qi] = idx
            request_completion_s[qi] = request_finished - retrieval_start
            request_search_service_s[qi] = request_finished - request_started
        else:
            futures = [
                search_executor.submit(search_one, local_i, qi)
                for local_i, qi in enumerate(faiss_indices)
            ]
            for f in as_completed(futures):
                qi, score, idx, request_started, request_finished = f.result()
                scores[qi] = score
                idxs[qi] = idx
                request_completion_s[qi] = request_finished - retrieval_start
                request_search_service_s[qi] = request_finished - request_started
        cpu_time_ed = time.perf_counter()
        phase_times["ivf_search_s"] = cpu_time_ed - cpu_time_st
        if batch_full_search:
            debug(
                f"FAISS batch-full fast path: {len(faiss_indices)} samples"
            )
        debug(f"CPU IVF search time: {phase_times['ivf_search_s']:.4f}s")

        # load docs
        flat_idxs = [d for sub in idxs if sub for d in sub]
        load_st = time.perf_counter()
        docs = load_docs(self.corpus, flat_idxs)
        load_ed = time.perf_counter()
        phase_times["load_docs_s"] = load_ed - load_st

        results = [docs[i * num : (i + 1) * num] for i in range(B)]

        # update query pool (batch)
        update_st = time.perf_counter()
        if query_pool is not None:
            batch_query_states = []
            retain_query_metadata = bool(
                getattr(query_pool, "retain_query_metadata", False)
            )
            for i, q in enumerate(query):
                if idxs[i] is None or actions[i] == "reuse":
                    continue
                topk_cluster_ids = self._clusters_for_doc_ids(idxs[i])
                batch_query_states.append(
                    QueryState(
                        query_id=(str(uuid.uuid4()) if retain_query_metadata else None),
                        query_text=(
                            q
                            if (
                                retain_query_metadata
                                or getattr(query_pool, "enable_exact_lookup", False)
                            )
                            else None
                        ),
                        embedding=emb[i],
                        topk_doc_ids=idxs[i],
                        topk_scores=scores[i],
                        topk_cluster_ids=topk_cluster_ids,
                    )
                )
            if batch_query_states:
                query_pool.add_queries_batch(batch_query_states)
        update_ed = time.perf_counter()
        phase_times["query_base_update_s"] = update_ed - update_st

        retrieval_end = time.perf_counter()
        self.last_retrieval_elapsed = retrieval_end - retrieval_start
        publish_request_timings(request_completion_s, request_search_service_s, actions)
        phase_times["total_s"] = self.last_retrieval_elapsed
        self._record_retrieval_profile(phase_times, actions, guided_probe_metrics)
        debug(
            f"Retrieval time for {B} samples: "
            f"{self.last_retrieval_elapsed:.4f}s"
        )

        if return_idx:
            return results, scores, idxs
        if return_score:
            return results, scores
        return results
