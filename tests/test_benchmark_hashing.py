import unittest

from benchmarks.benchmark_point import canonical_sha256


class BenchmarkHashingTest(unittest.TestCase):
    def test_digest_is_mapping_order_invariant_and_value_sensitive(self):
        reference = canonical_sha256({"pred": "a", "metrics": {"f1": 1.0}})
        reordered = canonical_sha256({"metrics": {"f1": 1.0}, "pred": "a"})
        changed = canonical_sha256({"pred": "b", "metrics": {"f1": 1.0}})

        self.assertEqual(reference, reordered)
        self.assertNotEqual(reference, changed)


if __name__ == "__main__":
    unittest.main()
