import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import benchmark_point
from benchmarks.benchmark_point import (
    DEFAULT_CPU_KV_CACHE_GB,
    DEFAULT_CPU_KV_CACHE_GIB,
    derive_cpu_kv_layout,
)


class BenchmarkCpuKvTest(unittest.TestCase):
    def test_cpu_kv_offload_defaults_to_fixed_capacity(self):
        self.assertEqual(DEFAULT_CPU_KV_CACHE_GB, 256.0)
        self.assertAlmostEqual(
            DEFAULT_CPU_KV_CACHE_GIB,
            256_000_000_000 / 1024**3,
        )

    def test_exact_block_count_overrides_capacity_derivation(self):
        with __import__("tempfile").TemporaryDirectory() as root:
            asset_root = __import__("pathlib").Path(root)
            model_dir = asset_root / "models/llama3-8B-instruct"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 4096,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "num_hidden_layers": 32,
                        "torch_dtype": "bfloat16",
                    }
                )
            )
            args = SimpleNamespace(
                cpu_kv_num_blocks=130_000,
                cpu_kv_cache_gib=1.0,
                kv_cache_dtype="fp8",
                cpu_kv_block_size=16,
                cpu_kv_eviction_policy="lru",
                cpu_kv_batch_h2d_transfers=True,
                cpu_kv_batch_d2h_transfers=True,
                cpu_kv_h2d_max_group_requests=0,
                cpu_kv_h2d_scheduling="async",
                cpu_kv_document_prefix_only=True,
                cpu_kv_min_document_heat_to_store=1.0,
                cpu_kv_h2d_stream_priority=-1,
                cpu_kv_d2h_stream_priority=0,
                cpu_kv_host_memory_mode="hybrid_pinned_lazy",
                cpu_kv_pinned_tier_gib=0.5,
                cpu_kv_lazy_mmap_dir=asset_root,
                cpu_kv_staging_buffer_blocks=1024,
            )

            with (
                patch.object(benchmark_point, "ASSET_ROOT", asset_root),
                patch.object(
                    benchmark_point.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=1 << 50),
                ),
            ):
                layout = derive_cpu_kv_layout(args)

        self.assertEqual(layout["num_cpu_blocks"], 130_000)
        self.assertEqual(layout["capacity_source"], "exact_block_count")
        self.assertEqual(layout["block_bytes"], 2**20)
        self.assertEqual(layout["dtype_bytes"], 1)
        self.assertEqual(layout["capacity_gib"], 130_000 * 2**20 / 1024**3)
        self.assertEqual(layout["lazy_allocation"], "file_mmap")
        self.assertEqual(layout["lazy_mmap_dir"], str(asset_root))


if __name__ == "__main__":
    unittest.main()
