import argparse
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.retrieval_cache_point import (
    DirectEngine,
    FaissBackend,
    load_questions,
)


class FakeSearchParametersIVF:
    def __init__(self):
        self.nprobe = 0


class FakeFaiss:
    SearchParametersIVF = FakeSearchParametersIVF


class FakeIndex:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def search(self, vectors, k, params=None):
        vectors = np.asarray(vectors)
        with self.lock:
            self.calls.append((vectors.shape, params.nprobe))
        values = vectors[:, :1].astype(np.int64)
        ids = np.repeat(values, k, axis=1)
        scores = ids.astype(np.float32)
        return scores, ids


class FakeQVCache:
    def __init__(self, ids):
        self.ids = ids

    def search(self, embedding, topk):
        return False, self.ids, np.zeros(topk, dtype=np.float32)


class FakeBackendIndex:
    def search(self, query, topk):
        return (
            np.asarray([[0.75, 0.50]], dtype=np.float32),
            np.asarray([[7, 8]], dtype=np.int64),
        )


def make_engine(executor):
    engine = object.__new__(DirectEngine)
    engine.args = argparse.Namespace(nprobe=128)
    engine.faiss = FakeFaiss()
    engine.index = FakeIndex()
    engine.search_executor = executor
    return engine


class RetrievalCacheParallelismTest(unittest.TestCase):
    def test_mixed_repeat_workload_warms_then_interleaves_hits_and_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            rows = [{"question": f"question-{index}"} for index in range(20)]
            (dataset_dir / "test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch(
                "benchmarks.retrieval_cache_point.DATASET_DIR",
                dataset_dir,
            ):
                questions, indices, source_count, _ = load_questions(
                    16,
                    7,
                    "test",
                    workload_mode="mixed_repeat",
                    unique_queries=4,
                )

        self.assertEqual(source_count, 20)
        self.assertEqual(len(questions), 16)
        self.assertEqual(len(set(indices)), 10)
        warm = indices[:4]
        mixed = indices[4:]
        self.assertEqual(mixed[::2], [warm[index % 4] for index in range(6)])
        self.assertTrue(all(index not in warm for index in mixed[1::2]))
        self.assertEqual(len(set(mixed[1::2])), 6)

    def test_query_parallel_preserves_input_order_and_per_call_nprobe(self):
        executor = ThreadPoolExecutor(max_workers=2)
        engine = make_engine(executor)
        try:
            _, ids = engine._search_faiss(
                np.asarray([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
                2,
            )
        finally:
            engine.close()

        np.testing.assert_array_equal(ids[:, 0], [3, 1, 2])
        self.assertEqual(sorted(engine.index.calls), [((1, 2), 128)] * 3)

    def test_native_batch_uses_one_call(self):
        engine = make_engine(None)
        _, ids = engine._search_faiss(
            np.asarray([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            2,
            nprobe=64,
        )

        np.testing.assert_array_equal(ids[:, 0], [3, 1, 2])
        self.assertEqual(engine.index.calls, [((3, 2), 64)])

    def test_qvcache_converts_document_ids_from_one_based(self):
        engine = object.__new__(DirectEngine)
        engine.args = argparse.Namespace(topk=2)
        engine.qvcache = FakeQVCache([1, 3])
        engine.corpus = [object(), object(), object()]
        engine.cache_hits = 0
        engine.cache_misses = 0

        ids = engine._search_qvcache(np.zeros((1, 2), dtype=np.float32))

        np.testing.assert_array_equal(ids, [[0, 2]])
        self.assertEqual(engine.cache_misses, 1)

    def test_qvcache_rejects_invalid_one_based_document_ids(self):
        engine = object.__new__(DirectEngine)
        engine.args = argparse.Namespace(topk=2)
        engine.qvcache = FakeQVCache([0, 3])
        engine.corpus = [object(), object(), object()]
        engine.cache_hits = 0
        engine.cache_misses = 0

        with self.assertRaisesRegex(RuntimeError, "one-based document ID"):
            engine._search_qvcache(np.zeros((1, 2), dtype=np.float32))

    def test_qvcache_l2_backend_converts_normalized_inner_products(self):
        backend = FaissBackend(FakeBackendIndex(), vectors=None, metric="l2")

        ids, distances = backend.search(np.asarray([1.0, 0.0]), 2)

        np.testing.assert_array_equal(ids, [7, 8])
        np.testing.assert_allclose(distances, [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
