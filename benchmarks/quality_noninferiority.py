"""Paired request-level quality analysis for the formal E2E campaign."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

try:
    from .fair_workload import array_sha256, file_sha256
except ImportError:
    from fair_workload import array_sha256, file_sha256


METRICS = ("retrieval_recall_at_k", "f1", "acc")
DEFAULT_MAXIMUM_DROPS = {
    "retrieval_recall_at_k": 0.005,
    "f1": 0.01,
    "acc": 0.01,
}


def load_request_metrics(summary: Mapping) -> dict[str, np.ndarray]:
    artifact = summary.get("request_metrics_artifact") or {}
    path = Path(artifact["path"])
    if file_sha256(path) != artifact.get("sha256"):
        raise ValueError(f"request metric artifact digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            "sample_indices": np.asarray(archive["sample_indices"], dtype="<i8"),
            **{metric: np.asarray(archive[metric], dtype="<f8") for metric in METRICS},
        }
    count = int(summary["num_requests"])
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError(f"request metric artifact shape mismatch: {path}")
    expected_sample_digest = (summary.get("arrival_trace_manifest") or {}).get(
        "sample_indices_sha256"
    )
    if array_sha256(arrays["sample_indices"]) != expected_sample_digest:
        raise ValueError(f"request metric sample order mismatch: {path}")
    for metric in METRICS:
        if not np.all(np.isfinite(arrays[metric])):
            raise ValueError(f"non-finite request metric {metric}: {path}")
    return arrays


def paired_bootstrap_mean_ci(
    differences,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype="<f8")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired differences must be a nonempty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be finite")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = np.random.default_rng(int(seed))
    means = np.empty(bootstrap_samples, dtype="<f8")
    chunk = 256
    for offset in range(0, bootstrap_samples, chunk):
        count = min(chunk, bootstrap_samples - offset)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "paired_requests": int(values.size),
        "confidence": float(confidence),
        "bootstrap_samples": int(bootstrap_samples),
    }


def _point_key(summary: Mapping) -> tuple:
    return (
        str(summary["dataset"]),
        str(summary["workflow"]),
        int(summary["nprobe"]),
        float(summary["arrival_rate_rps"]),
    )


def quality_noninferiority_report(
    summaries: Iterable[Mapping],
    *,
    maximum_drops: Mapping[str, float] = DEFAULT_MAXIMUM_DROPS,
    bootstrap_samples: int = 10_000,
    seed: int = 20261017,
) -> list[dict]:
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[_point_key(summary)].append(summary)
    output = []
    for point_offset, (point, rows) in enumerate(sorted(grouped.items())):
        by_repeat_system = {
            (int(row["campaign_repeat"]), str(row["system"])): row for row in rows
        }
        repeats = sorted({repeat for repeat, _system in by_repeat_system})
        matched_repeats = {
            baseline: [
                repeat
                for repeat in repeats
                if (repeat, "mnemorag") in by_repeat_system
                and (repeat, baseline) in by_repeat_system
            ]
            for baseline in ("flashrag", "hedrarag")
        }
        if not any(matched_repeats.values()):
            continue
        point_report = {
            "dataset": point[0],
            "workflow": point[1],
            "nprobe": point[2],
            "arrival_rate_rps": point[3],
            "repeats": repeats,
            "comparisons": {},
            "passes": True,
        }
        for baseline_offset, baseline in enumerate(("flashrag", "hedrarag")):
            if not matched_repeats[baseline]:
                continue
            differences = {metric: [] for metric in METRICS}
            for repeat in matched_repeats[baseline]:
                mnemo = load_request_metrics(by_repeat_system[(repeat, "mnemorag")])
                reference = load_request_metrics(by_repeat_system[(repeat, baseline)])
                if not np.array_equal(
                    mnemo["sample_indices"], reference["sample_indices"]
                ):
                    raise ValueError(
                        f"sample order differs for {point}, repeat {repeat}, {baseline}"
                    )
                for metric in METRICS:
                    differences[metric].append(mnemo[metric] - reference[metric])
            comparison = {"metrics": {}, "passes": True}
            comparison["matched_repeats"] = matched_repeats[baseline]
            for metric_offset, metric in enumerate(METRICS):
                result = paired_bootstrap_mean_ci(
                    np.concatenate(differences[metric]),
                    seed=(
                        seed
                        + 1000 * point_offset
                        + 100 * baseline_offset
                        + metric_offset
                    ),
                    bootstrap_samples=bootstrap_samples,
                )
                maximum_drop = float(maximum_drops[metric])
                result["maximum_allowed_drop"] = maximum_drop
                result["passes"] = result["ci_low"] >= -maximum_drop
                comparison["metrics"][metric] = result
                comparison["passes"] = comparison["passes"] and result["passes"]
            point_report["comparisons"][baseline] = comparison
            point_report["passes"] = point_report["passes"] and comparison["passes"]
        output.append(point_report)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("summaries_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    summaries = json.loads(args.summaries_json.read_text(encoding="utf-8"))
    report = quality_noninferiority_report(
        summaries, bootstrap_samples=args.bootstrap_samples
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)
    print(
        json.dumps(
            {"points": len(report), "all_pass": all(r["passes"] for r in report)}
        )
    )


if __name__ == "__main__":
    main()
