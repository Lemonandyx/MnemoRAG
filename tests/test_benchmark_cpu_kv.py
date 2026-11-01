import unittest

from benchmarks.benchmark_point import (
    DEFAULT_CPU_KV_CACHE_GB,
    DEFAULT_CPU_KV_CACHE_GIB,
)


class BenchmarkCpuKvTest(unittest.TestCase):
    def test_cpu_kv_offload_defaults_to_fixed_capacity(self):
        self.assertEqual(DEFAULT_CPU_KV_CACHE_GB, 256.0)
        self.assertAlmostEqual(
            DEFAULT_CPU_KV_CACHE_GIB,
            256_000_000_000 / 1024**3,
        )


if __name__ == "__main__":
    unittest.main()
