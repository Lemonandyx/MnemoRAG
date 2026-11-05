"""Fail-closed workload identity checks for manuscript experiments.

Request count alone is not a workload identity.  In particular, the formal
QPS proxy and the document-ordering stress workload both contain 512 NQ rows,
but they answer different experimental questions.  This module binds a named
role to the selection schema, ordered sample, online warmup, and dataset.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .fair_workload import (
        array_sha256,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
    )
except ImportError:
    from fair_workload import (
        array_sha256,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
    )


FORMAL_QPS_PROXY = "formal_qps_proxy"
DOCUMENT_ORDERING_STRESS = "document_ordering_stress"
WORKLOAD_ROLES = (FORMAL_QPS_PROXY, DOCUMENT_ORDERING_STRESS)

FORMAL_SELECTION_FORMAT = "mnemorag.prefix-equivalent-subset.v4"
FORMAL_SELECTION_SEMANTICS = (
    "causal online warmup followed by a disjoint 512-request measured "
    "segment matching complete-trace SAR and ordered-document-prefix state"
)
ORDERING_STRESS_FORMAT = "mnemorag.hint-scheme1-reordered512.v1"
ORDERING_STRESS_RULE = (
    "fixed-prior hotness order differs from retrieval order at each of "
    "top-k 3, 5, and 10"
)


class WorkloadIdentityError(ValueError):
    """Raised when an artifact does not satisfy its declared workload role."""


def _load_indices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype="<i8")
    if values.ndim != 1:
        raise WorkloadIdentityError(f"indices must be one-dimensional: {path}")
    return np.ascontiguousarray(values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkloadIdentityError(message)


def _resolve_artifact(
    manifest_path: Path,
    explicit_path: Path | None,
    sibling_name: str,
) -> Path:
    path = (
        Path(explicit_path)
        if explicit_path is not None
        else manifest_path.parent / sibling_name
    )
    _require(path.is_file(), f"missing workload artifact: {path}")
    return path.resolve()


def _formal_qps_identity(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    sample_indices_path: Path | None,
    warmup_indices_path: Path | None,
) -> dict[str, Any]:
    _require(
        manifest.get("format") == FORMAL_SELECTION_FORMAT,
        "formal_qps_proxy requires selection format "
        f"{FORMAL_SELECTION_FORMAT!r}; observed "
        f"{manifest.get('format') or manifest.get('schema')!r}",
    )
    _require(
        manifest.get("selection_semantics") == FORMAL_SELECTION_SEMANTICS,
        "formal QPS selection semantics do not match the audited contract",
    )
    _require(str(manifest.get("dataset") or "") != "", "selection dataset is missing")
    selected_rows = int(manifest.get("selected_rows", -1))
    warmup_rows = int(manifest.get("online_warmup_rows", -1))
    _require(selected_rows == 512, "formal QPS proxy must contain exactly 512 rows")
    _require(
        warmup_rows == selected_rows,
        "formal QPS proxy must use a separate 512-request online warmup",
    )

    prefix = manifest.get("prefix_contract") or {}
    expected_prefix = {
        "topk": 3,
        "document_hotness_freeze_prior": False,
        "cpu_kv_min_document_heat_to_store": 1,
        "measurement_excludes_warmup": True,
        "warmup_and_measurement_disjoint": True,
        "runtime_validation_required": True,
    }
    for key, expected in expected_prefix.items():
        _require(
            prefix.get(key) == expected,
            f"formal QPS prefix contract mismatch for {key}: "
            f"expected {expected!r}, observed {prefix.get(key)!r}",
        )

    validation = manifest.get("validation") or {}
    for key in (
        "offline_acceptance",
        "unique_selected_rows",
        "prefix_distribution_within_tolerance",
        "online_action_mix_within_tolerance",
        "hot_request_fraction_within_tolerance",
        "h2d_proxy_within_tolerance",
        "reuse_distance_distribution_within_tolerance",
        "unique_warmup_rows",
    ):
        _require(validation.get(key) is True, f"selection validation failed: {key}")
    for key in (
        "frozen_history_overlap_rows",
        "warmup_frozen_history_overlap_rows",
        "warmup_measurement_overlap_rows",
    ):
        _require(
            int(validation.get(key, -1)) == 0,
            f"selection overlap is nonzero: {key}",
        )

    sample_path = _resolve_artifact(
        manifest_path, sample_indices_path, "sample_indices.npy"
    )
    warmup_path = _resolve_artifact(
        manifest_path, warmup_indices_path, "warmup_indices.npy"
    )
    sample = _load_indices(sample_path)
    warmup = _load_indices(warmup_path)
    _require(
        sample.shape == (selected_rows,),
        "formal sample count does not match manifest",
    )
    _require(
        warmup.shape == (warmup_rows,),
        "formal warmup count does not match manifest",
    )
    _require(np.unique(sample).size == sample.size, "formal sample contains duplicates")
    _require(np.unique(warmup).size == warmup.size, "formal warmup contains duplicates")
    _require(
        np.intersect1d(sample, warmup).size == 0,
        "formal measurement and warmup overlap",
    )

    artifacts = manifest.get("artifacts") or {}
    sample_file_digest = file_sha256(sample_path)
    sample_array_digest = array_sha256(sample)
    warmup_file_digest = file_sha256(warmup_path)
    warmup_array_digest = array_sha256(warmup)
    expected_digests = {
        "sample_indices_sha256": sample_file_digest,
        "sample_indices_array_sha256": sample_array_digest,
        "warmup_indices_sha256": warmup_file_digest,
        "warmup_indices_array_sha256": warmup_array_digest,
    }
    for key, observed in expected_digests.items():
        _require(
            artifacts.get(key) == observed,
            f"selection artifact digest mismatch for {key}",
        )
    dataset_digest = str(artifacts.get("dataset_sha256") or "")
    _require(dataset_digest != "", "selection dataset digest is missing")

    return {
        "role": FORMAL_QPS_PROXY,
        "dataset": str(manifest["dataset"]),
        "selection_format": FORMAL_SELECTION_FORMAT,
        "selection_semantics": FORMAL_SELECTION_SEMANTICS,
        "selection_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": dataset_digest,
        "measurement_rows": selected_rows,
        "online_warmup_rows": warmup_rows,
        "sample_indices_file_sha256": sample_file_digest,
        "sample_indices_array_sha256": sample_array_digest,
        "warmup_indices_file_sha256": warmup_file_digest,
        "warmup_indices_array_sha256": warmup_array_digest,
        "topk": 3,
        "measurement_excludes_warmup": True,
    }


def _ordering_stress_identity(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    sample_indices_path: Path | None,
) -> dict[str, Any]:
    observed_format = manifest.get("schema") or manifest.get("format")
    _require(
        observed_format == ORDERING_STRESS_FORMAT,
        "document_ordering_stress requires selection schema "
        f"{ORDERING_STRESS_FORMAT!r}; observed {observed_format!r}",
    )
    _require(
        manifest.get("selection_rule") == ORDERING_STRESS_RULE,
        "document-ordering selection rule does not match the audited contract",
    )
    _require(
        manifest.get("selection_uses_quality_labels") is False,
        "stress selection uses quality labels",
    )
    selected_rows = int(manifest.get("selected_rows", -1))
    _require(
        selected_rows == 512,
        "document-ordering stress workload must have 512 rows",
    )
    changed = manifest.get("per_topk_changed_rows") or {}
    for topk in (3, 5, 10):
        _require(
            int(changed.get(str(topk), 0)) > 0,
            f"no ordering changes at top-k {topk}",
        )

    sample_path = _resolve_artifact(
        manifest_path, sample_indices_path, "selected_indices.npy"
    )
    sample = _load_indices(sample_path)
    _require(
        sample.shape == (selected_rows,),
        "stress sample count does not match manifest",
    )
    _require(np.unique(sample).size == sample.size, "stress sample contains duplicates")
    sample_digest = array_sha256(sample)
    _require(
        manifest.get("selected_indices_sha256") == sample_digest,
        "stress sample digest does not match manifest",
    )
    dataset_digest = str(manifest.get("dataset_sha256") or "")
    _require(dataset_digest != "", "stress dataset digest is missing")
    return {
        "role": DOCUMENT_ORDERING_STRESS,
        "dataset": str(manifest.get("dataset") or ""),
        "selection_format": ORDERING_STRESS_FORMAT,
        "selection_rule": ORDERING_STRESS_RULE,
        "selection_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": dataset_digest,
        "measurement_rows": selected_rows,
        "online_warmup_rows": 0,
        "sample_indices_array_sha256": sample_digest,
    }


def audit_workload_identity(
    selection_manifest_path: Path,
    *,
    expected_role: str,
    sample_indices_path: Path | None = None,
    warmup_indices_path: Path | None = None,
    arrival_trace_path: Path | None = None,
    expected_dataset: str | None = None,
    expected_request_count: int | None = None,
) -> dict[str, Any]:
    """Return a canonical identity or raise on any role/artifact mismatch."""
    if expected_role not in WORKLOAD_ROLES:
        raise WorkloadIdentityError(f"unsupported workload role: {expected_role}")
    manifest_path = Path(selection_manifest_path).resolve()
    _require(manifest_path.is_file(), f"missing selection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "selection manifest must be a JSON object")
    if expected_role == FORMAL_QPS_PROXY:
        identity = _formal_qps_identity(
            manifest_path,
            manifest,
            sample_indices_path=sample_indices_path,
            warmup_indices_path=warmup_indices_path,
        )
    else:
        identity = _ordering_stress_identity(
            manifest_path,
            manifest,
            sample_indices_path=sample_indices_path,
        )

    if expected_dataset is not None:
        _require(
            identity["dataset"] == expected_dataset,
            "dataset mismatch: expected "
            f"{expected_dataset}, observed {identity['dataset']}",
        )
    if expected_request_count is not None:
        _require(
            identity["measurement_rows"] == int(expected_request_count),
            "measurement request count does not match the declared workload",
        )

    if arrival_trace_path is not None:
        _, trace = load_trace_bundle(arrival_trace_path)
        trace_indices = load_trace_sample_indices(arrival_trace_path)
        _require(
            int((trace.get("spec") or {}).get("request_count", -1))
            == identity["measurement_rows"],
            "arrival-trace request count does not match workload identity",
        )
        _require(
            trace.get("dataset_sha256") == identity["dataset_sha256"],
            "arrival-trace dataset digest does not match workload identity",
        )
        _require(
            array_sha256(trace_indices) == identity["sample_indices_array_sha256"],
            "arrival-trace sample order does not match workload identity",
        )
        identity["arrival_trace_sha256"] = trace["arrival_trace_sha256"]
    return identity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-role", choices=WORKLOAD_ROLES, required=True)
    parser.add_argument("--sample-indices", type=Path)
    parser.add_argument("--warmup-indices", type=Path)
    parser.add_argument("--arrival-trace", type=Path)
    parser.add_argument("--expected-dataset")
    parser.add_argument("--expected-request-count", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        identity = audit_workload_identity(
            args.selection_manifest,
            expected_role=args.expected_role,
            sample_indices_path=args.sample_indices,
            warmup_indices_path=args.warmup_indices,
            arrival_trace_path=args.arrival_trace,
            expected_dataset=args.expected_dataset,
            expected_request_count=args.expected_request_count,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"workload identity audit failed: {error}") from error
    payload = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
