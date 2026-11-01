import numpy as np

from mnemorag.pipeline.utils import DocumentPool


def test_dense_counts_preserve_exact_hotness_and_ordering():
    pool = DocumentPool(num_documents=8)
    pool.update([1, 1, 4, 4, 4, 7])

    docs = [{"id": 1}, {"id": 4}, {"id": 7}]
    reordered_docs, reordered_ids = pool.reorder_by_hotness([1, 4, 7], docs)

    assert pool.get_count(1) == 2
    assert pool.get_count(np.int64(4)) == 3
    assert pool.total_docs() == 3
    assert reordered_ids == [4, 1, 7]
    assert reordered_docs == [{"id": 4}, {"id": 1}, {"id": 7}]


def test_dense_counts_use_fixed_width_storage_and_fallback_ids():
    pool = DocumentPool(num_documents=10)
    pool.update([2, 2, "external", 15])

    assert pool.dense_count_bytes == 10 * np.dtype(np.uint32).itemsize
    assert pool.get_count(2) == 2
    assert pool.get_count("external") == 1
    assert pool.get_count(15) == 1
    assert pool.total_docs() == 3


def test_topk_combines_dense_and_fallback_counts():
    pool = DocumentPool(num_documents=6)
    pool.update([1, 1, 1, 3, 3, "external", "external", "external", "external"])

    assert pool.topk(3) == [("external", 4), (1, 3), (3, 2)]


def test_dense_counts_preserve_duplicates_in_large_updates():
    pool = DocumentPool(num_documents=1_000)
    updates = list(range(300)) + [7, 7, 299]

    pool.update(updates)

    assert pool.get_count(7) == 3
    assert pool.get_count(299) == 2
    assert pool.total_docs() == 300


def test_dense_counts_saturate_instead_of_wrapping():
    pool = DocumentPool(num_documents=4)
    pool.dense_counts[2] = np.iinfo(np.uint32).max
    pool._dense_nonzero = 1

    pool.update([2, 2])

    assert pool.get_count(2) == np.iinfo(np.uint32).max
    assert pool.total_docs() == 1
