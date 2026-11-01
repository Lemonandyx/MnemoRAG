import unittest

from mnemorag.retriever.retriever import cache_manager, rerank_manager


class FakeCachedRetriever:
    def __init__(self, cache=None, use_cache=True, save_cache=False):
        self.cache = cache or {}
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.topk = 2
        self.miss_calls = []

    @cache_manager
    def batch_search(
        self,
        query=None,
        num=None,
        query_pool=None,
        return_score=False,
        return_idx=False,
    ):
        results = [[{"id": "7", "contents": "fresh"}] for _ in query]
        scores = [[0.7] for _ in query]
        idxs = [[7] for _ in query]
        return results, scores, idxs

    def _batch_search_with_rerank(
        self,
        query=None,
        num=None,
        query_pool=None,
        return_score=False,
        return_idx=False,
    ):
        self.miss_calls.append(
            {
                "query": query,
                "num": num,
                "query_pool": query_pool,
                "return_score": return_score,
                "return_idx": return_idx,
            }
        )
        results = [[{"id": "9", "contents": text}] for text in query]
        return results, [[0.9] for _ in query], [[9] for _ in query]


class FakeSingleCachedRetriever(FakeCachedRetriever):
    @cache_manager
    def search(
        self,
        query=None,
        num=None,
        query_pool=None,
        return_score=False,
        return_idx=False,
    ):
        return [{"id": "7", "contents": "fresh"}], [0.7], [7]


class ReverseReranker:
    def rerank(self, _query, results):
        return [list(reversed(row)) for row in results], [[0.9, 0.8]]


class FakeRerankRetriever:
    use_reranker = True
    reranker = ReverseReranker()

    @rerank_manager
    def batch_search(
        self,
        query=None,
        num=None,
        query_pool=None,
        return_score=False,
        return_idx=False,
    ):
        return (
            [[{"id": "1", "contents": "a"}, {"id": "2", "contents": "b"}]],
            [[0.2, 0.1]],
            [[1, 2]],
        )


class RetrievalCacheTest(unittest.TestCase):
    def test_legacy_cache_recovers_numeric_document_index(self):
        retriever = FakeCachedRetriever(
            cache={"cached": [{"id": "5", "contents": "doc", "score": 0.5}]}
        )

        results, scores, idxs = retriever.batch_search(
            ["cached"], return_score=True, return_idx=True
        )

        self.assertEqual(results, [[{"id": "5", "contents": "doc"}]])
        self.assertEqual(scores, [[0.5]])
        self.assertEqual(idxs, [[5]])

    def test_mixed_cache_passes_query_pool_by_keyword_and_preserves_order(self):
        query_pool = object()
        retriever = FakeCachedRetriever(
            cache={
                "cached": [
                    {
                        "id": "5",
                        "contents": "cached-doc",
                        "_retrieval_score": 0.5,
                        "_retrieval_idx": 5,
                    }
                ]
            }
        )

        results, scores, idxs = retriever.batch_search(
            ["cached", "missing"],
            query_pool=query_pool,
            return_score=True,
            return_idx=True,
        )

        self.assertEqual(
            [row[0]["contents"] for row in results], ["cached-doc", "missing"]
        )
        self.assertEqual(scores, [[0.5], [0.9]])
        self.assertEqual(idxs, [[5], [9]])
        self.assertEqual(
            retriever.miss_calls,
            [
                {
                    "query": ["missing"],
                    "num": 2,
                    "query_pool": query_pool,
                    "return_score": True,
                    "return_idx": True,
                }
            ],
        )

    def test_saved_cache_does_not_mutate_returned_documents(self):
        retriever = FakeCachedRetriever(use_cache=False, save_cache=True)

        results, scores, idxs = retriever.batch_search(
            ["fresh"], return_score=True, return_idx=True
        )

        self.assertEqual(results, [[{"id": "7", "contents": "fresh"}]])
        self.assertEqual(scores, [[0.7]])
        self.assertEqual(idxs, [[7]])
        self.assertEqual(
            retriever.cache["fresh"],
            [
                {
                    "id": "7",
                    "contents": "fresh",
                    "_retrieval_score": 0.7,
                    "_retrieval_idx": 7,
                }
            ],
        )

    def test_nonnumeric_legacy_id_requires_cache_regeneration(self):
        retriever = FakeCachedRetriever(
            cache={"cached": [{"id": "wiki-A", "contents": "doc", "score": 0.5}]}
        )

        with self.assertRaisesRegex(ValueError, "no Faiss document index"):
            retriever.batch_search(["cached"], return_idx=True)

    def test_single_query_cache_preserves_single_result_shape(self):
        retriever = FakeSingleCachedRetriever(
            cache={
                "cached": [
                    {
                        "id": "5",
                        "contents": "doc",
                        "_retrieval_score": 0.5,
                        "_retrieval_idx": 5,
                    }
                ]
            }
        )

        results, scores, idxs = retriever.search(
            "cached", return_score=True, return_idx=True
        )

        self.assertEqual(results, [{"id": "5", "contents": "doc"}])
        self.assertEqual(scores, [0.5])
        self.assertEqual(idxs, [5])

    def test_reranker_keeps_document_ids_aligned(self):
        retriever = FakeRerankRetriever()

        results, scores, idxs = retriever.batch_search(
            ["query"], return_score=True, return_idx=True
        )

        self.assertEqual([doc["id"] for doc in results[0]], ["2", "1"])
        self.assertEqual(scores, [[0.9, 0.8]])
        self.assertEqual(idxs, [[2, 1]])


if __name__ == "__main__":
    unittest.main()
