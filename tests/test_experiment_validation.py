import unittest

import numpy as np
import pytest

from benchmarks.quality import exact_retrieval_recall, recall_at_k

from benchmarks.experiment_validation import (
    expected_prompt_contract,
    verified_complete,
    verified_evaluation_contract,
    verified_point_contract,
    verified_prompt_contract,
    verified_quality_contract,
)


def test_recall_counts_exact_neighbors_instead_of_answer_strings():
    assert recall_at_k([[1, 2, 3], [1, 2, 3]], [[3, 2, 1], [4, 5, 3]], 3) == [
        1.0,
        1.0 / 3.0,
    ]
    with pytest.raises(ValueError, match="matching shapes"):
        recall_at_k([[1, 2, 3]], [[1, 2, 3], [1, 2, 3]], 3)
    with pytest.raises(ValueError, match="valid document"):
        recall_at_k([[1, -1, 3]], [[1, 2, 3]], 3)


def test_exact_reference_searches_all_lists_without_changing_runtime_budget():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(256, 8)).astype(np.float32)
    queries = rng.normal(size=(2, 8)).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(8), 8, 4, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 1
    reference = faiss.IndexFlatIP(8)
    reference.add(vectors)
    _, neighbors = reference.search(queries, 3)

    class Encoder:
        def encode(self, questions, **_):
            assert questions == ["q0", "q1"]
            return queries

    assert exact_retrieval_recall(Encoder(), index, ["q0", "q1"], neighbors, 3) == [
        1.0,
        1.0,
    ]
    assert index.nprobe == 1


def test_slo_boundary_uses_mean_latency_and_accepts_stopped_scan():
    from benchmarks.slo_campaign import (
        QPS_GRIDS,
        SYSTEMS,
        WORKFLOWS,
        select_boundary_qps,
    )

    rows = [
        {
            "workflow": workflow,
            "nprobe": 128,
            "system": system,
            "arrival_rate_rps": qps,
            "request_latency_mean_s": 2.0 if position < 2 else 11.0,
            "request_latency_p99_s": 30.0,
        }
        for workflow in WORKFLOWS
        for system in SYSTEMS
        for position, qps in enumerate(QPS_GRIDS[workflow][:3])
    ]
    selected = select_boundary_qps(rows, nprobes=(128,))
    for workflow in WORKFLOWS:
        assert selected[(workflow, 128)] == QPS_GRIDS[workflow][:3]


class ExperimentValidationTest(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "status": "complete",
            "metric": 0.0,
            "cpu_topology": {
                "physical_core_count": 64,
                "smt_siblings_selected": False,
            },
            "resource_preflight": {
                "clean_required": True,
                "violations": [],
            },
        }

    def test_accepts_verified_result_and_zero_valued_required_field(self):
        self.assertTrue(verified_complete(self.summary, 64, ("metric",)))

    def test_point_contract_rejects_missing_or_mismatched_workload_fields(self):
        summary = {
            "dataset": "nq",
            "workflow": "onepass",
            "nprobe": 128,
            "arrival_rate_rps": 24.0,
        }
        expected = dict(summary)
        self.assertTrue(verified_point_contract(summary, expected))
        self.assertFalse(verified_point_contract(summary, expected | {"nprobe": 512}))
        self.assertFalse(
            verified_point_contract(summary, expected | {"seed": 20260923})
        )

        complete = self.summary | summary
        self.assertTrue(
            verified_complete(
                complete,
                64,
                ("metric",),
                expected_point=expected,
            )
        )
        self.assertFalse(
            verified_complete(
                complete,
                64,
                ("metric",),
                expected_point=expected | {"workflow": "irg"},
            )
        )

    def test_rejects_unverified_or_incomplete_result(self):
        for mutation in (
            lambda item: item.update(status="failed"),
            lambda item: item["cpu_topology"].update(physical_core_count=63),
            lambda item: item["cpu_topology"].update(smt_siblings_selected=True),
            lambda item: item["resource_preflight"].update(clean_required=False),
            lambda item: item["resource_preflight"].update(violations=["busy"]),
            lambda item: item.pop("metric"),
        ):
            candidate = {
                **self.summary,
                "cpu_topology": dict(self.summary["cpu_topology"]),
                "resource_preflight": dict(self.summary["resource_preflight"]),
            }
            mutation(candidate)
            self.assertFalse(verified_complete(candidate, 64, ("metric",)))

    def test_quality_contract_requires_retrieval_recall_and_f1_acc(self):
        summary = {
            "num_requests": 32,
            "metrics": {
                "retrieval_recall_top5": 0.75,
                "f1": 0.50,
                "acc": 0.60,
            },
            "metric_observations": {
                "retrieval_recall_top5": 32,
                "f1": 32,
                "acc": 32,
            },
            "metric_contract": {
                "version": 3,
                "retrieval": {
                    "name": "exact_neighbor_recall_at_k",
                    "k": 5,
                    "scope": "final_retrieval_round",
                    "denominator": "all_requests",
                },
                "generation": ["f1", "acc"],
            },
        }
        self.assertTrue(verified_quality_contract(summary, 5))
        self.assertFalse(verified_quality_contract(summary, 3))
        summary["metrics"]["precision"] = 0.55
        self.assertFalse(verified_quality_contract(summary, 5))
        del summary["metrics"]["precision"]
        summary["metric_contract"]["generation"] = ["f1", "acc", "recall"]
        self.assertFalse(verified_quality_contract(summary, 5))
        summary["metric_contract"]["generation"] = ["f1", "acc"]
        summary["metric_contract"]["retrieval"]["denominator"] = "answered_only"
        self.assertFalse(verified_quality_contract(summary, 5))
        summary["metric_contract"]["retrieval"]["denominator"] = "all_requests"
        summary["metric_observations"]["f1"] = 31
        self.assertFalse(verified_quality_contract(summary, 5))

    def test_prompt_contract_requires_matched_templates_and_stage_budgets(self):
        summary = {"prompt_contract": expected_prompt_contract()}
        self.assertTrue(verified_prompt_contract(summary))
        summary["prompt_contract"]["recomp_compression_max_tokens"] = 32
        self.assertFalse(verified_prompt_contract(summary))

    def test_evaluation_contract_requires_quality_and_prompt_contracts(self):
        summary = {
            "num_requests": 32,
            "metrics": {
                "retrieval_recall_top5": 0.75,
                "f1": 0.50,
                "acc": 0.60,
            },
            "metric_observations": {
                "retrieval_recall_top5": 32,
                "f1": 32,
                "acc": 32,
            },
            "metric_contract": {
                "version": 3,
                "retrieval": {
                    "name": "exact_neighbor_recall_at_k",
                    "k": 5,
                    "scope": "final_retrieval_round",
                    "denominator": "all_requests",
                },
                "generation": ["f1", "acc"],
            },
            "prompt_contract": expected_prompt_contract(),
        }
        self.assertTrue(
            verified_evaluation_contract(summary, 5, ordering_mode="retrieval")
        )
        self.assertFalse(
            verified_evaluation_contract(summary, 5, ordering_mode="hotness_hint")
        )
        del summary["prompt_contract"]
        self.assertFalse(
            verified_evaluation_contract(summary, 5, ordering_mode="retrieval")
        )


if __name__ == "__main__":
    unittest.main()
