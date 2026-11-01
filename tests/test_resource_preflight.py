import argparse
import unittest
from unittest.mock import patch

from benchmarks import benchmark_point


class ResourcePreflightTest(unittest.TestCase):
    def test_smt_affinity_requires_explicit_opt_in(self):
        topology = "# CPU,Core,Socket,Node\n0,0,0,0\n128,0,0,0\n"
        with patch.object(
            benchmark_point.subprocess,
            "check_output",
            return_value=topology,
        ):
            with self.assertRaisesRegex(RuntimeError, "only 1 distinct"):
                benchmark_point.validate_physical_cpu_ids([0, 128])

            report = benchmark_point.validate_physical_cpu_ids([0, 128], allow_smt=True)

        self.assertEqual(report["logical_cpu_count"], 2)
        self.assertEqual(report["physical_core_count"], 1)
        self.assertTrue(report["smt_siblings_selected"])
        self.assertEqual(report["numa_node_ids"], [0])

    def test_selected_core_counters_take_precedence_over_global_load(self):
        args = argparse.Namespace(
            gpu_id="0",
            require_clean_resources=True,
            max_preflight_cpu_busy_percent=20.0,
            max_preflight_cpu_max_busy_percent=50.0,
            max_preflight_gpu_memory_mib=256.0,
            max_preflight_gpu_util_percent=10.0,
            max_preflight_load_per_cpu=0.1,
        )
        counters = [
            {3: (100, 90), 7: (100, 90)},
            {3: (200, 180), 7: (200, 180)},
        ]
        with (
            patch.object(
                benchmark_point,
                "_read_cpu_counters",
                side_effect=counters,
            ),
            patch.object(
                benchmark_point,
                "physical_sibling_cpu_ids",
                return_value=[3, 7],
            ),
            patch.object(benchmark_point.time, "sleep"),
            patch.object(
                benchmark_point.subprocess,
                "check_output",
                return_value="0, 0\n",
            ),
            patch.object(
                benchmark_point.os, "getloadavg", return_value=(4.0, 0.0, 0.0)
            ),
            patch.object(benchmark_point.os, "cpu_count", return_value=256),
        ):
            report = benchmark_point.resource_preflight(args, [3, 7])

        self.assertEqual(report["system_load_1m"], 4.0)
        self.assertEqual(report["selected_cpu_count"], 2)
        self.assertEqual(report["system_load_1m_per_cpu"], 2.0)
        self.assertTrue(report["selected_cpu_counters_complete"])
        self.assertEqual(report["violations"], [])

    def test_incomplete_selected_core_counters_are_rejected(self):
        args = argparse.Namespace(
            gpu_id="0",
            require_clean_resources=True,
            max_preflight_cpu_busy_percent=20.0,
            max_preflight_cpu_max_busy_percent=50.0,
            max_preflight_gpu_memory_mib=256.0,
            max_preflight_gpu_util_percent=10.0,
            max_preflight_load_per_cpu=0.1,
        )
        with (
            patch.object(
                benchmark_point,
                "_read_cpu_counters",
                side_effect=({3: (100, 90)}, {3: (200, 180)}),
            ),
            patch.object(
                benchmark_point,
                "physical_sibling_cpu_ids",
                return_value=[3, 7],
            ),
            patch.object(benchmark_point.time, "sleep"),
            patch.object(
                benchmark_point.subprocess,
                "check_output",
                return_value="0, 0\n",
            ),
            patch.object(
                benchmark_point.os,
                "getloadavg",
                return_value=(4.0, 0.0, 0.0),
            ),
            self.assertRaisesRegex(RuntimeError, "counters are incomplete"),
        ):
            benchmark_point.resource_preflight(args, [3, 7])


if __name__ == "__main__":
    unittest.main()
