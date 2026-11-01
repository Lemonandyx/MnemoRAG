"""Canonical request-arrival traces for cross-system serving experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TRACE_FORMAT = "mnemorag.fair-arrival-trace.v1"


@dataclass(frozen=True)
class ArrivalTraceSpec:
    request_count: int
    request_rate: float
    burstiness: float = 1.0
    random_seed: int = 0
    normalize_duration: bool = True


def _canonical_float64(values: Iterable[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(list(values), dtype="<f8"))


def _canonical_int64(values: Iterable[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(list(values), dtype="<i8"))


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_arrival_offsets(spec: ArrivalTraceSpec) -> np.ndarray:
    """Generate the single arrival trace replayed by every compared system."""
    request_count = int(spec.request_count)
    request_rate = float(spec.request_rate)
    burstiness = float(spec.burstiness)
    if request_count < 0:
        raise ValueError("request_count must be nonnegative")
    if request_count == 0:
        return np.asarray([], dtype="<f8")
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")
    if np.isinf(request_rate):
        return np.zeros(request_count, dtype="<f8")

    if np.isinf(burstiness):
        intervals = np.full(request_count, 1.0 / request_rate, dtype="<f8")
    else:
        rng = np.random.default_rng(int(spec.random_seed))
        intervals = rng.gamma(
            shape=burstiness,
            scale=1.0 / (request_rate * burstiness),
            size=request_count,
        )
    offsets = np.cumsum(intervals, dtype="<f8")
    if spec.normalize_duration and offsets[-1] > 0:
        offsets *= (request_count / request_rate) / offsets[-1]
    validate_arrival_offsets(offsets, request_count=request_count)
    return offsets


def validate_arrival_offsets(
    offsets: Iterable[float], *, request_count: int | None = None
) -> np.ndarray:
    canonical = _canonical_float64(offsets)
    if request_count is not None and canonical.shape != (int(request_count),):
        raise ValueError(
            f"arrival trace has {canonical.size} values; expected {request_count}"
        )
    if not np.all(np.isfinite(canonical)):
        raise ValueError("arrival trace contains non-finite values")
    if np.any(canonical < 0) or np.any(np.diff(canonical) < 0):
        raise ValueError("arrival offsets must be nonnegative and monotonic")
    return canonical


def build_trace_manifest(
    spec: ArrivalTraceSpec,
    offsets: Iterable[float],
    *,
    sample_indices: Iterable[int] | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    canonical = validate_arrival_offsets(offsets, request_count=spec.request_count)
    duration_s = float(canonical[-1]) if canonical.size else 0.0
    manifest: dict[str, Any] = {
        "format": TRACE_FORMAT,
        "spec": asdict(spec),
        "arrival_trace_sha256": array_sha256(canonical),
        "duration_s": duration_s,
        "realized_request_rate": (
            float(spec.request_count) / duration_s if duration_s > 0 else None
        ),
    }
    if sample_indices is not None:
        indices = _canonical_int64(sample_indices)
        if indices.shape != (int(spec.request_count),):
            raise ValueError(
                f"sample_indices has {indices.size} values; expected {spec.request_count}"
            )
        manifest["sample_indices_sha256"] = array_sha256(indices)
        manifest["sample_indices_count"] = int(indices.size)
    if dataset_sha256 is not None:
        manifest["dataset_sha256"] = str(dataset_sha256)
    return manifest


def validate_trace_workload(
    manifest: dict[str, Any],
    spec: ArrivalTraceSpec,
    *,
    sample_indices: Iterable[int],
    dataset_sha256: str,
) -> None:
    """Reject a trace that was not built for the exact evaluated workload."""
    if manifest.get("spec") != asdict(spec):
        raise ValueError("arrival trace specification does not match this point")
    observed_sample_digest = array_sha256(_canonical_int64(sample_indices))
    if manifest.get("sample_indices_sha256") != observed_sample_digest:
        raise ValueError("arrival trace sample order does not match this point")
    if manifest.get("dataset_sha256") != str(dataset_sha256):
        raise ValueError("arrival trace dataset digest does not match this point")


def summarize_latency_slo(
    latencies_s: Iterable[float], *, wall_time_s: float, slo_seconds: float
) -> dict[str, float | int]:
    latencies = _canonical_float64(latencies_s)
    if latencies.size == 0:
        raise ValueError("latencies_s must not be empty")
    if not np.all(np.isfinite(latencies)) or np.any(latencies < 0):
        raise ValueError("latencies_s must be finite and nonnegative")
    if wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")
    if slo_seconds <= 0:
        raise ValueError("slo_seconds must be positive")
    attained = int(np.count_nonzero(latencies <= float(slo_seconds)))
    return {
        "slo_attained_requests": attained,
        "slo_total_requests": int(latencies.size),
        "slo_attainment": attained / int(latencies.size),
        "slo_goodput_qps": attained / float(wall_time_s),
    }


def write_trace_bundle(
    path: Path,
    spec: ArrivalTraceSpec,
    offsets: Iterable[float],
    *,
    sample_indices: Iterable[int] | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically persist a trace and the fingerprints used for result audits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_arrival_offsets(offsets, request_count=spec.request_count)
    manifest = build_trace_manifest(
        spec,
        canonical,
        sample_indices=sample_indices,
        dataset_sha256=dataset_sha256,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        indices = (
            _canonical_int64(sample_indices)
            if sample_indices is not None
            else np.asarray([], dtype="<i8")
        )
        np.savez_compressed(
            stream,
            arrival_offsets=canonical,
            sample_indices=indices,
            manifest_json=np.asarray(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            ),
        )
    temporary.replace(path)
    return manifest


def load_trace_bundle(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        offsets = validate_arrival_offsets(bundle["arrival_offsets"])
        manifest = json.loads(str(bundle["manifest_json"].item()))
    if manifest.get("format") != TRACE_FORMAT:
        raise ValueError(f"unsupported arrival-trace format: {manifest.get('format')}")
    expected = manifest.get("arrival_trace_sha256")
    actual = array_sha256(offsets)
    if actual != expected:
        raise ValueError(
            f"arrival-trace digest mismatch: expected {expected}, observed {actual}"
        )
    expected_count = int((manifest.get("spec") or {})["request_count"])
    validate_arrival_offsets(offsets, request_count=expected_count)
    return offsets, manifest


def load_trace_sample_indices(path: Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "sample_indices" not in bundle:
            raise ValueError("arrival-trace bundle does not contain sample indices")
        indices = _canonical_int64(bundle["sample_indices"])
        manifest = json.loads(str(bundle["manifest_json"].item()))
    expected_count = int(manifest.get("sample_indices_count", -1))
    if indices.shape != (expected_count,):
        raise ValueError(
            f"sample index count mismatch: expected {expected_count}, "
            f"observed {indices.size}"
        )
    expected_digest = manifest.get("sample_indices_sha256")
    observed_digest = array_sha256(indices)
    if expected_digest != observed_digest:
        raise ValueError(
            "sample index digest mismatch: "
            f"expected {expected_digest}, observed {observed_digest}"
        )
    return indices


def select_sample_indices(
    source_count: int,
    sample_count: int,
    *,
    random_seed: int,
    excluded_indices: Iterable[int] = (),
) -> np.ndarray:
    """Select a deterministic sample, optionally excluding historical rows."""
    source_count = int(source_count)
    sample_count = int(sample_count)
    if source_count < 0 or sample_count < 0:
        raise ValueError("source_count and sample_count must be nonnegative")
    excluded = set(map(int, excluded_indices))
    if any(index < 0 or index >= source_count for index in excluded):
        raise ValueError("excluded sample index is outside the dataset")
    candidates = np.asarray(
        [index for index in range(source_count) if index not in excluded],
        dtype="<i8",
    )
    if sample_count > candidates.size:
        raise ValueError(
            f"requested {sample_count} rows from {candidates.size} available"
        )
    rng = np.random.default_rng(int(random_seed))
    selected = rng.choice(candidates, size=sample_count, replace=False)
    # Request order is part of a cross-request caching workload. Preserve the
    # seeded permutation instead of converting it into dataset-source order.
    return selected.astype("<i8", copy=False)
