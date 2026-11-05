"""Regression tests for workload-role confusion in manuscript experiments."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.comparison_validation import (
    ComparisonContractError,
    validate_result_workload_identity,
)
from benchmarks.fair_workload import (
    ArrivalTraceSpec,
    array_sha256,
    build_arrival_offsets,
    file_sha256,
    validate_trace_workload,
    write_trace_bundle,
)
from benchmarks.workload_identity import (
    DOCUMENT_ORDERING_STRESS,
    FORMAL_QPS_PROXY,
    FORMAL_SELECTION_FORMAT,
    FORMAL_SELECTION_SEMANTICS,
    ORDERING_STRESS_FORMAT,
    ORDERING_STRESS_RULE,
    WorkloadIdentityError,
    audit_workload_identity,
)


class WorkloadIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sample = np.arange(512, dtype="<i8")
        self.warmup = np.arange(512, 1024, dtype="<i8")
        self.sample_path = self.root / "sample_indices.npy"
        self.warmup_path = self.root / "warmup_indices.npy"
        np.save(self.sample_path, self.sample, allow_pickle=False)
        np.save(self.warmup_path, self.warmup, allow_pickle=False)

    def tearDown(self):
        self.temporary.cleanup()

    def _formal_manifest(self) -> Path:
        manifest = {
            "format": FORMAL_SELECTION_FORMAT,
            "dataset": "nq",
            "selected_rows": 512,
            "online_warmup_rows": 512,
            "selection_semantics": FORMAL_SELECTION_SEMANTICS,
            "prefix_contract": {
                "topk": 3,
                "document_hotness_freeze_prior": False,
                "cpu_kv_min_document_heat_to_store": 1,
                "measurement_excludes_warmup": True,
                "warmup_and_measurement_disjoint": True,
                "runtime_validation_required": True,
            },
            "validation": {
                "offline_acceptance": True,
                "unique_selected_rows": True,
                "prefix_distribution_within_tolerance": True,
                "online_action_mix_within_tolerance": True,
                "hot_request_fraction_within_tolerance": True,
                "h2d_proxy_within_tolerance": True,
                "reuse_distance_distribution_within_tolerance": True,
                "unique_warmup_rows": True,
                "frozen_history_overlap_rows": 0,
                "warmup_frozen_history_overlap_rows": 0,
                "warmup_measurement_overlap_rows": 0,
            },
            "artifacts": {
                "dataset_sha256": "dataset-digest",
                "sample_indices_sha256": file_sha256(self.sample_path),
                "sample_indices_array_sha256": array_sha256(self.sample),
                "warmup_indices_sha256": file_sha256(self.warmup_path),
                "warmup_indices_array_sha256": array_sha256(self.warmup),
            },
        }
        path = self.root / "selection_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _stress_manifest(self) -> Path:
        selected_path = self.root / "selected_indices.npy"
        np.save(selected_path, self.sample, allow_pickle=False)
        manifest = {
            "schema": ORDERING_STRESS_FORMAT,
            "dataset": "nq",
            "dataset_sha256": "dataset-digest",
            "selected_rows": 512,
            "selected_indices_sha256": array_sha256(self.sample),
            "selection_rule": ORDERING_STRESS_RULE,
            "selection_uses_quality_labels": False,
            "per_topk_changed_rows": {"3": 10, "5": 20, "10": 30},
        }
        path = self.root / "stress_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_accepts_formal_qps_proxy_with_bound_warmup(self):
        identity = audit_workload_identity(
            self._formal_manifest(),
            expected_role=FORMAL_QPS_PROXY,
            expected_dataset="nq",
            expected_request_count=512,
        )
        self.assertEqual(identity["role"], FORMAL_QPS_PROXY)
        self.assertEqual(
            identity["sample_indices_array_sha256"], array_sha256(self.sample)
        )
        self.assertEqual(identity["online_warmup_rows"], 512)

    def test_rejects_ordering_stress_set_when_formal_qps_is_required(self):
        with self.assertRaisesRegex(
            WorkloadIdentityError, "formal_qps_proxy requires selection format"
        ):
            audit_workload_identity(
                self._stress_manifest(),
                expected_role=FORMAL_QPS_PROXY,
                sample_indices_path=self.root / "selected_indices.npy",
            )

    def test_accepts_ordering_stress_only_under_its_own_role(self):
        identity = audit_workload_identity(
            self._stress_manifest(),
            expected_role=DOCUMENT_ORDERING_STRESS,
            sample_indices_path=self.root / "selected_indices.npy",
        )
        self.assertEqual(identity["role"], DOCUMENT_ORDERING_STRESS)
        self.assertEqual(identity["online_warmup_rows"], 0)

    def test_rejects_tampered_warmup(self):
        manifest = self._formal_manifest()
        np.save(self.warmup_path, self.warmup[::-1], allow_pickle=False)
        with self.assertRaisesRegex(WorkloadIdentityError, "digest mismatch"):
            audit_workload_identity(manifest, expected_role=FORMAL_QPS_PROXY)

    def test_trace_manifest_binds_and_validates_workload_identity(self):
        identity = audit_workload_identity(
            self._formal_manifest(), expected_role=FORMAL_QPS_PROXY
        )
        spec = ArrivalTraceSpec(512, 8.0, random_seed=7)
        offsets = build_arrival_offsets(spec)
        trace_path = self.root / "workload.npz"
        manifest = write_trace_bundle(
            trace_path,
            spec,
            offsets,
            sample_indices=self.sample,
            dataset_sha256="dataset-digest",
            workload_identity=identity,
        )
        validate_trace_workload(
            manifest,
            spec,
            sample_indices=self.sample,
            dataset_sha256="dataset-digest",
            workload_identity=identity,
        )
        with self.assertRaisesRegex(ValueError, "workload identity"):
            validate_trace_workload(
                manifest,
                spec,
                sample_indices=self.sample,
                dataset_sha256="dataset-digest",
                workload_identity=identity | {"role": DOCUMENT_ORDERING_STRESS},
            )

    def test_result_audit_rejects_missing_or_wrong_workload_role(self):
        identity = audit_workload_identity(
            self._formal_manifest(), expected_role=FORMAL_QPS_PROXY
        )
        summary = {
            "system": "mnemorag",
            "dataset": "nq",
            "num_requests": 512,
            "arrival_trace_manifest": {
                "dataset_sha256": "dataset-digest",
                "sample_indices_sha256": array_sha256(self.sample),
                "workload_identity": identity,
            },
        }
        self.assertEqual(
            validate_result_workload_identity(
                summary, expected_role=FORMAL_QPS_PROXY
            ),
            identity,
        )
        with self.assertRaisesRegex(ComparisonContractError, "does not match"):
            validate_result_workload_identity(
                summary, expected_role=DOCUMENT_ORDERING_STRESS
            )
        del summary["arrival_trace_manifest"]["workload_identity"]
        with self.assertRaisesRegex(
            ComparisonContractError, "missing workload identity"
        ):
            validate_result_workload_identity(
                summary, expected_role=FORMAL_QPS_PROXY
            )


if __name__ == "__main__":
    unittest.main()
