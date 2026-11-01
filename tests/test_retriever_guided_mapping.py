import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import faiss
import numpy as np
import torch

from mnemorag.pipeline.utils import QueryPool, QueryState
from mnemorag.retriever.retriever import (
    DenseRetriever,
    _merge_guided_probe_lists,
    _resolve_similarity_thresholds,
)


class FakeEncoder:
    def __init__(self):
        self.last_queries = None

    def encode(self, queries, **_):
        self.last_queries = list(queries)
        return np.asarray(
            [[index] for index, _ in enumerate(queries)], dtype=np.float32
        )


class FakeQuantizer:
    def __init__(self):
        self.calls = 0

    def search(self, embeddings, nprobe):
        self.calls += 1
        rows = []
        for embedding in embeddings:
            base = (int(embedding[0]) + 1) * 10
            rows.append(np.arange(base, base + nprobe, dtype=np.int64))
        return np.zeros((len(rows), nprobe), dtype=np.float32), np.asarray(rows)

    def reconstruct_n(self, start, count):
        return np.arange(start, start + count, dtype=np.float32).reshape(-1, 1)


class FakeIVFIndex:
    def __init__(self):
        self.quantizer = FakeQuantizer()
        self.probes_by_query = {}
        self.distances_by_query = {}
        self.nprobes_by_query = {}
        self.nlist = 128
        self.metric_type = faiss.METRIC_INNER_PRODUCT

    def search(self, embedding, k, params=None):
        query_index = int(embedding[0, 0])
        if params is not None:
            self.nprobes_by_query[query_index] = params.nprobe
            if hasattr(params, "probes"):
                self.probes_by_query[query_index] = params.probes.tolist()
        return (
            np.ones((1, k), dtype=np.float32),
            np.arange(
                100 + query_index,
                100 + query_index + k,
                dtype=np.int64,
            ).reshape(1, -1),
        )


class FakeQueryPool:
    def __init__(self):
        self.min_similarity = 0.70
        self.max_similarity = 1.00
        self.query_states = [
            QueryState(
                query_id="history",
                query_text="history",
                embedding=None,
                topk_doc_ids=[900],
                topk_scores=[0.9],
                topk_cluster_ids=[99],
            )
        ]
        self.added = []

    def current_size(self):
        return 1

    def search_similar_batch(self, _embeddings, topk=1):
        return (
            torch.tensor([[0.99], [0.80], [0.80]]),
            torch.tensor([[0], [0], [0]]),
        )

    def touch(self, _slots):
        return None

    def add_queries_batch(self, states):
        self.added.extend(states)


class EmptyQueryPool(FakeQueryPool):
    def __init__(self):
        super().__init__()
        self.query_states = []

    def current_size(self):
        return 0

    def search_similar_batch(self, _embeddings, topk=1):
        raise AssertionError("Cold Query Base must bypass similarity search")


class UninitializedRangeQueryPool(FakeQueryPool):
    def __init__(self):
        super().__init__()
        self.min_similarity = 1.0
        self.max_similarity = 0.0

    def search_similar_batch(self, _embeddings, topk=1):
        return torch.tensor([[0.90]]), torch.tensor([[0]])


class AllReuseQueryPool(FakeQueryPool):
    def search_similar_batch(self, embeddings, topk=1):
        batch_size = len(embeddings)
        return (
            torch.full((batch_size, 1), 0.99),
            torch.zeros((batch_size, 1), dtype=torch.long),
        )


class FixedThresholdTest(unittest.TestCase):
    def test_production_thresholds_ignore_unapproved_overrides(self):
        config = {
            "similarity_t_reuse": 0.90,
            "similarity_t_guided": 0.60,
        }
        with patch.dict(
            os.environ,
            {
                "MNEMORAG_T_REUSE": "0.91",
                "MNEMORAG_T_GUIDED": "0.61",
            },
            clear=False,
        ):
            os.environ.pop("MNEMORAG_ALLOW_THRESHOLD_OVERRIDE", None)
            self.assertEqual(_resolve_similarity_thresholds(config), (0.95, 0.85))


class GuidedProbeMergeTest(unittest.TestCase):
    def test_partial_budget_appends_only_novel_historical_lists(self):
        self.assertEqual(
            _merge_guided_probe_lists(
                [7, 3, 7],
                [3, 9, 11],
                allow_historical_expansion=True,
            ),
            [7, 3, 9, 11],
        )

    def test_full_budget_preserves_standard_query_lists(self):
        self.assertEqual(
            _merge_guided_probe_lists(
                [7, 3, 7],
                [3, 9, 11],
                allow_historical_expansion=False,
            ),
            [7, 3],
        )


class PartitionedTopKTest(unittest.TestCase):
    def test_single_query_partitioning_is_opt_in(self):
        retriever = object.__new__(DenseRetriever)
        retriever._config = {}
        self.assertEqual(retriever._single_query_probe_partition_limit(), 1)

        retriever._config = {"enable_single_query_probe_parallelism": True}
        self.assertEqual(retriever._single_query_probe_partition_limit(), 8)

    def test_inner_product_merge_matches_global_topk(self):
        partitions = [
            (
                np.asarray([[0.9, 0.7, 0.1]], dtype=np.float32),
                np.asarray([[9, 7, 1]], dtype=np.int64),
            ),
            (
                np.asarray([[0.8, 0.6, 0.2]], dtype=np.float32),
                np.asarray([[8, 6, 2]], dtype=np.int64),
            ),
        ]

        distances, labels = DenseRetriever._merge_partitioned_topk(
            partitions, 3, faiss.METRIC_INNER_PRODUCT
        )

        np.testing.assert_allclose(distances, [[0.9, 0.8, 0.7]])
        np.testing.assert_array_equal(labels, [[9, 8, 7]])

    def test_l2_merge_ignores_invalid_candidates(self):
        partitions = [
            (
                np.asarray([[0.1, 0.5, np.inf]], dtype=np.float32),
                np.asarray([[1, 5, -1]], dtype=np.int64),
            ),
            (
                np.asarray([[0.2, 0.3, 0.9]], dtype=np.float32),
                np.asarray([[2, 3, 9]], dtype=np.int64),
            ),
        ]

        distances, labels = DenseRetriever._merge_partitioned_topk(
            partitions, 3, faiss.METRIC_L2
        )

        np.testing.assert_allclose(distances, [[0.1, 0.2, 0.3]])
        np.testing.assert_array_equal(labels, [[1, 2, 3]])

    def test_parallel_preassigned_search_matches_faiss_ivf(self):
        random = np.random.default_rng(17)
        vectors = random.normal(size=(1024, 16)).astype(np.float32)
        query = random.normal(size=(1, 16)).astype(np.float32)
        quantizer = faiss.IndexFlatIP(16)
        index = faiss.IndexIVFFlat(quantizer, 16, 32, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = 16
        coarse_distances, probes = index.quantizer.search(query, index.nprobe)
        expected_distances, expected_labels = index.search(query, 5)
        retriever = object.__new__(DenseRetriever)
        retriever._search_executor_workers = 4

        with ThreadPoolExecutor(max_workers=4) as executor:
            distances, labels = retriever._search_preassigned_parallel(
                index,
                query,
                5,
                probes[0],
                coarse_distances[0],
                executor,
                max_partitions=4,
            )

        np.testing.assert_allclose(distances, expected_distances, rtol=1e-6)
        np.testing.assert_array_equal(labels, expected_labels)


class GuidedClusterMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_threshold_override = os.environ.get(
            "MNEMORAG_ALLOW_THRESHOLD_OVERRIDE"
        )
        os.environ["MNEMORAG_ALLOW_THRESHOLD_OVERRIDE"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls._old_threshold_override is None:
            os.environ.pop("MNEMORAG_ALLOW_THRESHOLD_OVERRIDE", None)
        else:
            os.environ["MNEMORAG_ALLOW_THRESHOLD_OVERRIDE"] = (
                cls._old_threshold_override
            )

    def test_exact_query_fast_path_bypasses_encoder_and_faiss(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = Mock()
        retriever.encoder.encode.side_effect = AssertionError(
            "exact lookup must bypass encoding"
        )
        retriever.index = object()
        retriever.corpus = [{"id": "0", "contents": "cached"}]
        query_pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=2,
            enable_exact_lookup=True,
        )
        query_pool.add_queries_batch(
            [
                QueryState(
                    query_id=None,
                    query_text="repeat",
                    embedding=torch.tensor([1.0, 0.0]),
                    topk_doc_ids=[0],
                    topk_scores=[0.9],
                    topk_cluster_ids=[3],
                )
            ]
        )

        results, scores, indices = retriever._batch_search(
            ["repeat"], query_pool=query_pool, return_idx=True
        )

        retriever.close()
        self.assertEqual(results, [[{"id": "0", "contents": "cached"}]])
        self.assertAlmostEqual(scores[0][0], 0.9, places=6)
        self.assertEqual(indices, [[0]])
        self.assertEqual(
            retriever.retrieval_profile()["exact_query_fast_path"],
            {"batches": 1, "mixed_batches": 0, "requests": 1},
        )

    def test_mixed_exact_batch_encodes_and_searches_only_misses(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 10}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "enable_preassigned_guidance": True,
            "ivf_nprobe": 128,
            "faiss_search_workers": 1,
        }
        query_pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=4,
            enable_exact_lookup=True,
        )
        query_pool.add_queries_batch(
            [
                QueryState(
                    query_id=None,
                    query_text="repeat",
                    embedding=torch.tensor([1.0]),
                    topk_doc_ids=[0],
                    topk_scores=[0.9],
                    topk_cluster_ids=[3],
                )
            ]
        )
        ivf_index = FakeIVFIndex()

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            results, scores, indices = retriever._batch_search(
                ["repeat", "new"],
                query_pool=query_pool,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(retriever.encoder.last_queries, ["new"])
        self.assertEqual(results, [["doc-0"], ["doc-100"]])
        self.assertAlmostEqual(scores[0][0], 0.9, places=6)
        self.assertEqual(indices, [[0], [100]])
        self.assertEqual(query_pool.lookup_exact_batch(["repeat", "new"]), [0, 1])
        profile = retriever.retrieval_profile()
        self.assertEqual(
            profile["exact_query_fast_path"],
            {"batches": 1, "mixed_batches": 1, "requests": 1},
        )
        self.assertEqual(profile["action_counts"]["exact"], 1)
        self.assertEqual(retriever.last_similarity_action_counts["exact"], 1)

    def test_adaptive_reuse_first_skips_quantizer_after_all_hit_batch(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "enable_preassigned_guidance": True,
            "ivf_nprobe": 128,
        }
        query_pool = AllReuseQueryPool()
        ivf_index = FakeIVFIndex()

        with (
            patch.dict(
                os.environ,
                {"MNEMORAG_ADAPTIVE_REUSE_FIRST_QUANTIZER": "1"},
            ),
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            first = retriever._batch_search(
                ["repeat"], query_pool=query_pool, return_idx=True
            )
            second = retriever._batch_search(
                ["repeat"], query_pool=query_pool, return_idx=True
            )

        retriever.close()
        self.assertEqual(first[2], [[900]])
        self.assertEqual(second[2], [[900]])
        self.assertEqual(ivf_index.quantizer.calls, 1)
        self.assertEqual(
            retriever.retrieval_profile()["adaptive_reuse_first"],
            {"batches": 1, "quantizer_skips": 1},
        )

    def test_reuse_payload_is_trimmed_to_requested_topk(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "ivf_nprobe": 128,
        }
        query_pool = AllReuseQueryPool()
        query_pool.query_states[0].topk_doc_ids = [900, 901]
        query_pool.query_states[0].topk_scores = [0.9, 0.8]
        ivf_index = FakeIVFIndex()

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            results, scores, indices = retriever._batch_search(
                ["repeat"], num=1, query_pool=query_pool, return_idx=True
            )

        retriever.close()
        self.assertEqual(indices, [[900]])
        self.assertEqual(scores, [[0.9]])
        self.assertEqual(results, [["doc-900"]])
        self.assertEqual(retriever.retrieval_profile()["action_counts"], {"reuse": 1})

    def test_incomplete_reuse_payload_falls_back_to_full_ivf(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 2
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200, 101: 201}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "ivf_nprobe": 128,
        }
        query_pool = AllReuseQueryPool()
        ivf_index = FakeIVFIndex()
        ivf_index.ntotal = 1000

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            results, scores, indices = retriever._batch_search(
                ["deeper"], num=2, query_pool=query_pool, return_idx=True
            )

        retriever.close()
        self.assertEqual(indices, [[100, 101]])
        self.assertEqual(scores, [[1.0, 1.0]])
        self.assertEqual(results, [["doc-100", "doc-101"]])
        self.assertEqual(retriever.retrieval_profile()["action_counts"], {"full": 1})

    def test_first_guided_batch_uses_threshold_normalization(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "enable_preassigned_guidance": True,
            "ivf_nprobe": 128,
        }
        query_pool = UninitializedRangeQueryPool()
        ivf_index = FakeIVFIndex()

        def fake_preassigned(_index, embedding, _k, probes, _distances):
            query_index = int(embedding[0, 0])
            ivf_index.probes_by_query[query_index] = probes.tolist()
            return (
                np.asarray([[1.0]], dtype=np.float32),
                np.asarray([[100]], dtype=np.int64),
            )

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
            patch.object(
                DenseRetriever,
                "_search_preassigned",
                side_effect=fake_preassigned,
            ),
        ):
            _results, _scores, indices = retriever._batch_search(
                ["guided"],
                query_pool=query_pool,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(indices, [[100]])
        guided = retriever.retrieval_profile()["guided_probe_stats"]
        self.assertEqual(guided["probe_budget_histogram"], {"18": 1})
        self.assertAlmostEqual(query_pool.min_similarity, 0.90, places=6)
        self.assertAlmostEqual(query_pool.max_similarity, 0.90, places=6)

    def test_guided_search_falls_back_when_restricted_lists_are_incomplete(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {101: 201}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "enable_preassigned_guidance": True,
            "ivf_nprobe": 128,
        }
        query_pool = FakeQueryPool()
        ivf_index = FakeIVFIndex()
        ivf_index.ntotal = 1000

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
            patch.object(
                DenseRetriever,
                "_search_preassigned",
                return_value=(
                    np.asarray([[-np.inf]], dtype=np.float32),
                    np.asarray([[-1]], dtype=np.int64),
                ),
            ),
        ):
            _results, _scores, indices = retriever._batch_search(
                ["reuse", "guided"],
                query_pool=query_pool,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(indices, [[900], [101]])
        self.assertEqual(
            retriever.retrieval_profile()["guided_probe_stats"][
                "incomplete_result_fallbacks"
            ],
            1,
        )

    def test_exact_queries_are_coalesced_within_a_batch(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200, 101: 201}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "ivf_nprobe": 128,
            "faiss_search_workers": 2,
        }
        query_pool = EmptyQueryPool()
        ivf_index = FakeIVFIndex()

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            results, scores, indices = retriever._batch_search(
                ["same", "same", "other"],
                query_pool=query_pool,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(retriever.encoder.last_queries, ["same", "other"])
        self.assertEqual(indices, [[100], [100], [101]])
        self.assertEqual(results, [["doc-100"], ["doc-100"], ["doc-101"]])
        self.assertEqual(scores, [[1.0], [1.0], [1.0]])
        self.assertEqual(len(query_pool.added), 2)
        self.assertEqual(
            retriever.retrieval_profile()["exact_duplicate_requests_coalesced"],
            1,
        )

    def test_mmap_index_load_preserves_ivf_results(self):
        rng = np.random.default_rng(11)
        vectors = rng.normal(size=(256, 8)).astype(np.float32)
        queries = rng.normal(size=(4, 8)).astype(np.float32)
        faiss.normalize_L2(vectors)
        faiss.normalize_L2(queries)
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(8), 8, 8, faiss.METRIC_INNER_PRODUCT
        )
        index.train(vectors)
        index.add(vectors)
        index.nprobe = 4
        expected_scores, expected_ids = index.search(queries, 3)

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "tiny.index")
            faiss.write_index(index, index_path)
            retriever = object.__new__(DenseRetriever)
            retriever.index_path = index_path
            retriever._config = {
                "faiss_mmap_index": True,
                "ivf_nprobe": 4,
            }
            retriever.initialization_profile = {}
            retriever.use_faiss_gpu = False
            retriever.load_index()
            actual_scores, actual_ids = retriever.index.search(queries, 3)

        self.assertTrue(retriever.initialization_profile["index_mmap"])
        self.assertTrue(np.array_equal(actual_ids, expected_ids))
        self.assertTrue(np.allclose(actual_scores, expected_scores))

    def test_reused_prefix_does_not_shift_guided_quantizer_rows(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {101: 201, 102: 202}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "ivf_nprobe": 128,
        }
        query_pool = FakeQueryPool()
        ivf_index = FakeIVFIndex()

        def fake_preassigned(_index, embedding, _k, probes, _distances):
            query_index = int(embedding[0, 0])
            ivf_index.probes_by_query[query_index] = probes.tolist()
            ivf_index.distances_by_query[query_index] = _distances.tolist()
            return (
                np.asarray([[1.0]], dtype=np.float32),
                np.asarray([[100 + query_index]], dtype=np.int64),
            )

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
            patch.object(
                DenseRetriever,
                "_search_preassigned",
                side_effect=fake_preassigned,
            ),
        ):
            _results, _scores, indices = retriever._batch_search(
                ["reuse", "guided-one", "guided-two"],
                query_pool=query_pool,
                return_score=True,
                return_idx=True,
            )

        retriever.close()

        self.assertEqual(indices[0], [900])
        self.assertIn(20, ivf_index.probes_by_query[1])
        self.assertNotIn(10, ivf_index.probes_by_query[1])
        self.assertIn(30, ivf_index.probes_by_query[2])
        self.assertNotIn(20, ivf_index.probes_by_query[2])
        self.assertTrue(np.allclose(ivf_index.distances_by_query[1][:-1], 0.0))
        self.assertAlmostEqual(ivf_index.distances_by_query[1][-1], 99.0)
        self.assertTrue(np.allclose(ivf_index.distances_by_query[2][:-1], 0.0))
        self.assertAlmostEqual(ivf_index.distances_by_query[2][-1], 198.0)
        self.assertEqual(len(query_pool.added), 2)
        self.assertEqual(ivf_index.quantizer.calls, 1)
        profile = retriever.retrieval_profile()
        guided = profile["guided_probe_stats"]
        self.assertEqual(guided["observations"], 2)
        self.assertEqual(guided["probe_budget_mean"], 34.0)
        self.assertEqual(guided["nprobe_mean"], 35.0)
        self.assertEqual(guided["historical_clusters_mean"], 1.0)
        self.assertEqual(guided["historical_novel_clusters_mean"], 1.0)
        self.assertEqual(guided["historical_added_clusters_mean"], 1.0)
        self.assertEqual(guided["probe_budget_histogram"], {"34": 2})
        self.assertEqual(guided["nprobe_histogram"], {"35": 2})
        retriever.reset_retrieval_profile()
        reset_profile = retriever.retrieval_profile()
        self.assertEqual(reset_profile["batches"], 0)
        self.assertEqual(reset_profile["action_counts"], {})
        self.assertEqual(reset_profile["guided_probe_stats"]["observations"], 0)

    def test_nprobe_only_ablation_keeps_the_query_probe_budget(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {101: 201, 102: 202}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "enable_preassigned_guidance": False,
            "ivf_nprobe": 128,
        }
        query_pool = FakeQueryPool()
        ivf_index = FakeIVFIndex()

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            retriever._batch_search(
                ["reuse", "guided-one", "guided-two"],
                query_pool=query_pool,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(ivf_index.nprobes_by_query, {1: 34, 2: 34})
        guided = retriever.retrieval_profile()["guided_probe_stats"]
        self.assertEqual(guided["probe_budget_mean"], 34.0)
        self.assertEqual(guided["nprobe_mean"], 34.0)
        self.assertEqual(guided["historical_added_clusters_mean"], 0.0)

    def test_cold_query_base_skips_unused_quantizer_search(self):
        retriever = object.__new__(DenseRetriever)
        retriever.topk = 1
        retriever.batch_size = 8
        retriever.encoder = FakeEncoder()
        retriever.index = object()
        retriever.corpus = object()
        retriever.docid_to_cluster = {100: 200, 101: 201}
        retriever._config = {
            "similarity_t_reuse": 0.95,
            "similarity_t_guided": 0.70,
            "enable_similarity_aware": True,
            "ivf_nprobe": 128,
            "faiss_search_workers": 2,
        }
        query_pool = EmptyQueryPool()
        ivf_index = FakeIVFIndex()

        with (
            patch(
                "mnemorag.retriever.retriever.faiss.extract_index_ivf",
                return_value=ivf_index,
            ),
            patch(
                "mnemorag.retriever.retriever.load_docs",
                side_effect=lambda _corpus, ids: [f"doc-{doc_id}" for doc_id in ids],
            ),
        ):
            _results, _scores, indices = retriever._batch_search(
                ["cold-one", "cold-two"],
                query_pool=query_pool,
                return_score=True,
                return_idx=True,
            )

        retriever.close()
        self.assertEqual(indices, [[100], [101]])
        self.assertEqual(ivf_index.quantizer.calls, 0)
        self.assertEqual(len(query_pool.added), 2)
        self.assertIn("ivf_search_s", retriever.last_retrieval_phase_times)
        profile = retriever.retrieval_profile()
        self.assertEqual(profile["batches"], 1)
        self.assertEqual(profile["action_counts"], {"full": 2})

    def test_preassigned_search_scans_the_supplied_ivf_list(self):
        rng = np.random.default_rng(7)
        dimension = 16
        vectors = rng.normal(size=(1000, dimension)).astype(np.float32)
        query = rng.normal(size=(1, dimension)).astype(np.float32)
        faiss.normalize_L2(vectors)
        faiss.normalize_L2(query)

        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            8,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(vectors)
        index.add(vectors)
        centroid_distances, centroid_ids = index.quantizer.search(query, 8)

        top_distances, top_ids = DenseRetriever._search_preassigned(
            index,
            query,
            5,
            centroid_ids[:, :1],
            centroid_distances[:, :1],
        )
        params = faiss.SearchParametersIVF()
        params.nprobe = 1
        expected_distances, expected_ids = index.search(query, 5, params=params)
        self.assertTrue(np.array_equal(top_ids, expected_ids))
        self.assertTrue(np.allclose(top_distances, expected_distances))

        capped_distances, capped_ids = DenseRetriever._search_preassigned(
            index,
            query,
            5,
            centroid_ids[:, :1],
            centroid_distances[:, :1],
            max_codes=16,
        )
        capped_params = faiss.SearchParametersIVF()
        capped_params.nprobe = 1
        capped_params.max_codes = 16
        expected_capped_distances, expected_capped_ids = index.search(
            query, 5, params=capped_params
        )
        self.assertTrue(np.array_equal(capped_ids, expected_capped_ids))
        self.assertTrue(np.allclose(capped_distances, expected_capped_distances))

        _, far_ids = DenseRetriever._search_preassigned(
            index,
            query,
            5,
            centroid_ids[:, -1:],
            centroid_distances[:, -1:],
        )
        self.assertFalse(np.array_equal(far_ids, expected_ids))


if __name__ == "__main__":
    unittest.main()
