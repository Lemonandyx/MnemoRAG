import os
import unittest
from unittest.mock import patch

from mnemorag.config.config import Config


class CpuConfigTest(unittest.TestCase):
    def call_init_cpu(self, values, topology=None, affinity=None):
        config = object.__new__(Config)
        config.final_config = dict(values)
        affinity = set(range(64)) if affinity is None else set(affinity)
        topology = topology or (lambda cpu_id: (0, cpu_id))
        with (
            patch(
                "mnemorag.config.config.os.sched_getaffinity",
                return_value=affinity,
                create=True,
            ),
            patch(
                "mnemorag.config.config.os.sched_setaffinity",
                create=True,
            ) as set_affinity,
            patch(
                "mnemorag.config.config._cpu_topology",
                side_effect=topology,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            config._init_cpu()
            environment = {
                name: os.environ[name]
                for name in (
                    "MNEMORAG_FAISS_OMP_THREADS",
                    "MNEMORAG_FAISS_SEARCH_WORKERS",
                    "MNEMORAG_FAISS_MAX_CODES",
                    "MNEMORAG_FAISS_BATCH_FULL",
                    "OMP_NUM_THREADS",
                )
            }
        return config.final_config, environment, set_affinity

    def test_default_selects_64_distinct_physical_cores(self):
        values, environment, set_affinity = self.call_init_cpu({})

        self.assertEqual(values["faiss_omp_threads"], 1)
        self.assertEqual(values["faiss_search_workers"], 64)
        self.assertEqual(values["physical_core_count"], 64)
        self.assertEqual(values["cpu_socket_ids"], [0])
        self.assertFalse(values["smt_siblings_selected"])
        self.assertEqual(environment["MNEMORAG_FAISS_OMP_THREADS"], "1")
        self.assertEqual(environment["MNEMORAG_FAISS_SEARCH_WORKERS"], "64")
        self.assertEqual(environment["MNEMORAG_FAISS_MAX_CODES"], "0")
        self.assertEqual(environment["MNEMORAG_FAISS_BATCH_FULL"], "0")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        set_affinity.assert_called_once_with(0, set(range(64)))

    def test_one_thread_per_core_is_selected_from_smt_affinity(self):
        values, _, set_affinity = self.call_init_cpu(
            {},
            affinity=range(128),
            topology=lambda cpu_id: (0, cpu_id % 64),
        )

        self.assertEqual(values["cpu_affinity"], list(range(64)))
        set_affinity.assert_called_once_with(0, set(range(64)))

    def test_non_64_cpu_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cpu_threads=64"):
            self.call_init_cpu({"cpu_threads": 32})

    def test_non_64_worker_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "faiss_search_workers=64"):
            self.call_init_cpu({"faiss_search_workers": 8})

    def test_nested_openmp_parallelism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "faiss_omp_threads=1"):
            self.call_init_cpu({"faiss_omp_threads": 2})

    def test_requires_64_physical_cores_on_one_socket(self):
        with self.assertRaisesRegex(RuntimeError, "64 distinct physical"):
            self.call_init_cpu(
                {},
                affinity=range(64),
                topology=lambda cpu_id: (0, cpu_id % 32),
            )

    def test_max_codes_is_preserved(self):
        values, environment, _ = self.call_init_cpu(
            {
                "faiss_max_codes": 131072,
            }
        )

        self.assertEqual(values["faiss_max_codes"], 131072)
        self.assertEqual(environment["MNEMORAG_FAISS_MAX_CODES"], "131072")

    def test_negative_max_codes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "faiss_max_codes"):
            self.call_init_cpu({"faiss_max_codes": -1})


if __name__ == "__main__":
    unittest.main()
