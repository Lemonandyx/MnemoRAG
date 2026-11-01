import unittest
from unittest.mock import patch

import numpy as np

from mnemorag.retriever.retriever import DenseRetriever


class FakeInvertedLists:
    def __init__(self, lists):
        self.lists = [np.asarray(values, dtype=np.int64) for values in lists]

    def list_size(self, list_id):
        return len(self.lists[list_id])

    def get_ids(self, list_id):
        return self.lists[list_id]


class FakeIndex:
    def __init__(self, lists, ntotal):
        self.invlists = FakeInvertedLists(lists)
        self.nlist = len(lists)
        self.ntotal = ntotal


class DocIdClusterMappingTest(unittest.TestCase):
    def test_dense_ids_use_compact_int32_array(self):
        index = FakeIndex([[0, 2], [1, 3]], ntotal=4)
        with patch(
            "mnemorag.retriever.retriever.faiss.rev_swig_ptr",
            side_effect=lambda values, _size: values,
        ):
            mapping = DenseRetriever._build_docid_to_cluster(index)

        self.assertEqual(mapping.dtype, np.int32)
        self.assertEqual(mapping.tolist(), [0, 1, 0, 1])
        self.assertEqual(mapping.nbytes, 16)

    def test_sparse_ids_fall_back_to_dictionary(self):
        index = FakeIndex([[10], [30]], ntotal=2)
        with patch(
            "mnemorag.retriever.retriever.faiss.rev_swig_ptr",
            side_effect=lambda values, _size: values,
        ):
            mapping = DenseRetriever._build_docid_to_cluster(index)

        self.assertEqual(mapping, {10: 0, 30: 1})

    def test_cluster_lookup_ignores_invalid_dense_ids(self):
        retriever = object.__new__(DenseRetriever)
        retriever.docid_to_cluster = np.asarray([2, 4, -1], dtype=np.int32)

        self.assertEqual(retriever._clusters_for_doc_ids([0, 2, 8, -1, 1]), [2, 4])


if __name__ == "__main__":
    unittest.main()
