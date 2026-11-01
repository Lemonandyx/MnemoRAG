import tempfile
import unittest
from array import array
from pathlib import Path

import numpy as np
import torch

from mnemorag.pipeline.utils import QueryPool, QueryState


def make_state(query_id, embedding):
    return QueryState(
        query_id=query_id,
        query_text=query_id,
        embedding=embedding,
        topk_doc_ids=[0, 1, 2],
        topk_scores=[1.0, 0.5, 0.25],
        topk_cluster_ids=[0, 1, 2],
    )


class QueryPoolTest(unittest.TestCase):
    def test_search_matches_reference_cosine_similarity(self):
        pool = QueryPool(device="cpu", dtype="float32", max_size=4)
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        pool.add_queries_batch(
            [make_state(str(index), row) for index, row in enumerate(embeddings)]
        )

        queries = np.asarray([[0.9, 0.1], [-0.8, 0.2]], dtype=np.float32)
        scores, indices = pool.search_similar_batch(queries)
        reference = torch.nn.functional.normalize(torch.from_numpy(queries), dim=1)
        reference = (
            reference
            @ torch.nn.functional.normalize(torch.from_numpy(embeddings), dim=1).T
        )

        torch.testing.assert_close(scores[:, 0], reference.max(dim=1).values)
        self.assertEqual(indices[:, 0].tolist(), [0, 2])
        state = pool.query_states[0]
        self.assertIsNone(state.query_id)
        self.assertIsNone(state.query_text)
        self.assertIsNone(state.embedding)
        self.assertIsInstance(state.topk_doc_ids, array)
        self.assertIsInstance(state.topk_scores, array)
        self.assertIsInstance(state.topk_cluster_ids, array)

    def test_lru_reuses_fixed_slots(self):
        pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=3,
            retain_query_metadata=True,
        )
        pool.add_queries_batch(
            [
                make_state("a", torch.tensor([1.0, 0.0])),
                make_state("b", torch.tensor([0.0, 1.0])),
                make_state("c", torch.tensor([-1.0, 0.0])),
            ]
        )

        pool.touch([0])
        pool.add_queries_batch([make_state("d", torch.tensor([0.5, 0.5]))])

        self.assertEqual(pool.current_size(), 3)
        self.assertEqual(
            [state.query_id for state in pool.query_states], ["a", "d", "c"]
        )
        self.assertEqual(tuple(pool.embedding_matrix.shape), (3, 2))

    def test_search_retains_general_topk_path(self):
        pool = QueryPool(device="cpu", dtype="float32", max_size=3)
        pool.add_queries_batch(
            [
                make_state("a", torch.tensor([1.0, 0.0])),
                make_state("b", torch.tensor([0.8, 0.2])),
                make_state("c", torch.tensor([0.0, 1.0])),
            ]
        )

        scores, indices = pool.search_similar_batch(torch.tensor([[1.0, 0.0]]), topk=2)

        self.assertEqual(indices.tolist(), [[0, 1]])
        self.assertEqual(tuple(scores.shape), (1, 2))

    def test_exact_lookup_tracks_lru_slot_replacement(self):
        pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=2,
            enable_exact_lookup=True,
        )
        pool.add_queries_batch(
            [
                make_state("a", torch.tensor([1.0, 0.0])),
                make_state("b", torch.tensor([0.0, 1.0])),
            ]
        )

        self.assertEqual(pool.lookup_exact_batch(["a", "b", "missing"]), [0, 1, None])
        self.assertIsNone(pool.query_states[0].query_text)
        pool.touch([0])
        pool.add_queries_batch([make_state("c", torch.tensor([-1.0, 0.0]))])

        self.assertEqual(pool.lookup_exact_batch(["a", "b", "c"]), [0, None, 1])
        self.assertEqual(pool.current_size(), 2)

    def test_exact_lookup_upserts_existing_query_without_consuming_capacity(self):
        pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=3,
            enable_exact_lookup=True,
        )
        pool.add_queries_batch(
            [
                make_state("a", torch.tensor([1.0, 0.0])),
                make_state("b", torch.tensor([0.0, 1.0])),
            ]
        )
        replacement = make_state("a", torch.tensor([-1.0, 0.0]))
        replacement.topk_doc_ids = [10, 11, 12, 13]
        replacement.topk_scores = [1.0, 0.8, 0.6, 0.4]

        pool.add_queries_batch([replacement])

        self.assertEqual(pool.current_size(), 2)
        self.assertEqual(pool.lookup_exact_batch(["a", "b"]), [0, 1])
        self.assertEqual(list(pool.query_states[0].topk_doc_ids), [10, 11, 12, 13])
        torch.testing.assert_close(pool.embedding_matrix[0], torch.tensor([-1.0, 0.0]))
        pool.add_queries_batch([make_state("c", torch.tensor([0.5, 0.5]))])
        pool.add_queries_batch([make_state("d", torch.tensor([0.5, -0.5]))])
        self.assertEqual(pool.lookup_exact_batch(["a", "b", "c", "d"]), [0, None, 2, 1])

    def test_batch_overflow_keeps_capacity_and_normalized_storage(self):
        pool = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=2,
            retain_query_metadata=True,
        )
        pool.add_queries_batch(
            [
                make_state("a", np.asarray([3.0, 0.0], dtype=np.float32)),
                make_state("b", np.asarray([0.0, 4.0], dtype=np.float32)),
                make_state("c", np.asarray([-5.0, 0.0], dtype=np.float32)),
            ]
        )

        self.assertEqual(pool.current_size(), 2)
        torch.testing.assert_close(
            torch.linalg.vector_norm(pool.embedding_matrix[:2], dim=1),
            torch.ones(2),
        )
        scores, indices = pool.search_similar_batch(
            np.asarray([[-1.0, 0.0]], dtype=np.float32)
        )
        self.assertAlmostEqual(scores[0, 0].item(), 1.0, places=6)
        self.assertEqual(pool.query_states[indices[0, 0].item()].query_id, "c")

    def test_snapshot_round_trip_preserves_similarity_and_retrieval_state(self):
        source = QueryPool(
            device="cpu",
            dtype="float32",
            max_size=4,
            enable_exact_lookup=True,
        )
        source.add_queries_batch(
            [
                make_state("a", np.asarray([3.0, 0.0], dtype=np.float32)),
                make_state("b", np.asarray([0.0, 4.0], dtype=np.float32)),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_base.npz"
            metadata = source.save_snapshot(
                path, metadata={"dataset": "nq", "retrieval_topk": 3}
            )
            restored = QueryPool(
                device="cpu",
                dtype="float32",
                max_size=4,
                enable_exact_lookup=True,
            )
            loaded = restored.load_snapshot(path)

        self.assertEqual(metadata, loaded)
        self.assertEqual(restored.current_size(), 2)
        self.assertEqual(restored.lookup_exact_batch(["a", "b"]), [0, 1])
        self.assertEqual(list(restored.query_states[0].topk_doc_ids), [0, 1, 2])
        scores, indices = restored.search_similar_batch(
            np.asarray([[1.0, 0.0]], dtype=np.float32)
        )
        self.assertEqual(indices.tolist(), [[0]])
        self.assertAlmostEqual(scores[0, 0].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
