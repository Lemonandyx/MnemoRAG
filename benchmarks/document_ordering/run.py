#!/usr/bin/env python3
"""Run the formal Scheme-1 Hint validation on 512 always-reordered NQ rows.

The measured rows are selected without using answer-quality labels. Every row
changes order under the fixed document-hotness prior at top-k 3, 5, and 10.
All nine cells replay the same selected source rows and retrieval results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOPKS = (3, 5, 10)
MODES = ("original", "hotness", "hotness_hint")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
DISPLAY = {
    "original": "Original",
    "hotness": "Hotness",
    "hotness_hint": "Hotness+Hint",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hotness_order(ids: list[int], prior: dict[int, int]) -> list[int]:
    heat = [int(prior.get(document_id, 0)) for document_id in ids]
    if not heat or max(heat) <= 1:
        return list(ids)
    positions = sorted(
        range(len(ids)),
        key=lambda position: (-heat[position], position),
    )
    return [ids[position] for position in positions]


def prepare_workload(args: argparse.Namespace) -> dict[str, object]:
    sys.path.insert(0, str(args.project_root))
    from benchmarks.fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        write_trace_bundle,
    )

    retrieval_ids = np.load(
        args.retrieval_ids,
        mmap_mode="r",
        allow_pickle=False,
    )
    retrieval_scores = np.load(
        args.retrieval_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    history_indices = set(
        map(
            int,
            np.asarray(
                np.load(args.history_indices, allow_pickle=False),
                dtype=np.int64,
            ),
        )
    )
    prior = {
        int(key): int(value)
        for key, value in json.loads(
            args.hotness_prior.read_text(encoding="utf-8")
        ).items()
    }
    if retrieval_ids.shape != retrieval_scores.shape:
        raise ValueError("Retrieval ID and score caches have different shapes")
    if retrieval_ids.ndim != 2 or retrieval_ids.shape[1] < max(TOPKS):
        raise ValueError(f"Unexpected retrieval cache shape: {retrieval_ids.shape}")

    eligible: list[int] = []
    per_topk_counts = {topk: 0 for topk in TOPKS}
    for source_index, raw_row in enumerate(retrieval_ids):
        if source_index in history_indices:
            continue
        row = list(map(int, raw_row[: max(TOPKS)]))
        if len(set(row)) != len(row):
            continue
        changes = {}
        for topk in TOPKS:
            source = row[:topk]
            changes[topk] = hotness_order(source, prior) != source
            per_topk_counts[topk] += int(changes[topk])
        if all(changes.values()):
            eligible.append(source_index)
    if len(eligible) < args.num_requests:
        raise RuntimeError(
            f"Only {len(eligible)} rows reorder at every top-k; "
            f"need {args.num_requests}"
        )

    rng = np.random.default_rng(args.selection_seed)
    selected = rng.choice(
        np.asarray(eligible, dtype="<i8"),
        size=args.num_requests,
        replace=False,
    ).astype("<i8", copy=False)
    if len(set(map(int, selected))) != args.num_requests:
        raise RuntimeError("Selected source indices are not unique")

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_root / "selected_indices.npy"
    with selected_path.open("wb") as stream:
        np.save(stream, selected, allow_pickle=False)

    dataset_sha256 = file_sha256(args.dataset_path)
    spec = ArrivalTraceSpec(
        request_count=args.num_requests,
        request_rate=args.arrival_rate,
        burstiness=1.0,
        random_seed=args.arrival_seed,
        normalize_duration=True,
    )
    arrival_trace = args.output_root / "arrival_trace.npz"
    trace_manifest = write_trace_bundle(
        arrival_trace,
        spec,
        build_arrival_offsets(spec),
        sample_indices=selected,
        dataset_sha256=dataset_sha256,
    )

    reference_paths = {}
    for topk in TOPKS:
        reference_path = args.output_root / f"retrieval_reference_topk{topk}.csv"
        temporary = reference_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "request_index",
                    "source_index",
                    "retrieval_idxs",
                    "retrieval_scores",
                ),
            )
            writer.writeheader()
            for request_index, source_index in enumerate(map(int, selected)):
                writer.writerow(
                    {
                        "request_index": request_index,
                        "source_index": source_index,
                        "retrieval_idxs": json.dumps(
                            list(map(int, retrieval_ids[source_index, :topk])),
                            separators=(",", ":"),
                        ),
                        "retrieval_scores": json.dumps(
                            list(map(float, retrieval_scores[source_index, :topk])),
                            separators=(",", ":"),
                        ),
                    }
                )
        temporary.replace(reference_path)
        reference_paths[topk] = reference_path

    manifest = {
        "schema": "mnemorag.hint-scheme1-reordered512.v1",
        "dataset": "nq",
        "source_rows": int(retrieval_ids.shape[0]),
        "history_rows_excluded": len(history_indices),
        "selection_uses_quality_labels": False,
        "selection_rule": (
            "fixed-prior hotness order differs from retrieval order at each "
            "of top-k 3, 5, and 10"
        ),
        "eligible_rows": len(eligible),
        "per_topk_changed_rows": {
            str(key): value for key, value in per_topk_counts.items()
        },
        "selected_rows": args.num_requests,
        "selection_seed": args.selection_seed,
        "selected_indices_sha256": array_sha256(selected),
        "selected_indices_path": str(selected_path.resolve()),
        "arrival_trace_path": str(arrival_trace.resolve()),
        "arrival_trace_manifest": trace_manifest,
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "retrieval_ids_path": str(args.retrieval_ids.resolve()),
        "retrieval_ids_sha256": file_sha256(args.retrieval_ids),
        "retrieval_scores_path": str(args.retrieval_scores.resolve()),
        "retrieval_scores_sha256": file_sha256(args.retrieval_scores),
        "history_indices_path": str(args.history_indices.resolve()),
        "history_indices_sha256": file_sha256(args.history_indices),
        "hotness_prior_path": str(args.hotness_prior.resolve()),
        "hotness_prior_sha256": file_sha256(args.hotness_prior),
        "retrieval_references": {
            str(topk): {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for topk, path in reference_paths.items()
        },
        "prompt_contract": {
            "documents": "all retrieved top-k documents",
            "hotness_hint_order": "fixed document-hotness descending",
            "hint_position": "system-prompt tail after document context",
            "hint_content": "complete original relevance priority order",
            "inline_document_marker": False,
        },
    }
    write_json(args.output_root / "selection_manifest.json", manifest)
    return manifest


def point_dir(args: argparse.Namespace, topk: int, mode: str) -> Path:
    return args.output_root / "points" / f"topk{topk}" / mode


def summary_path(args: argparse.Namespace, topk: int, mode: str) -> Path:
    point_name = (
        f"nq_onepass_full_k{topk}_np128_rps{args.arrival_rate:g}_n{args.num_requests}"
    )
    return point_dir(args, topk, mode) / point_name / "summary.json"


def verified_point(
    args: argparse.Namespace,
    topk: int,
    mode: str,
    expected_indices: list[int],
) -> bool:
    path = summary_path(args, topk, mode)
    requests_path = path.parent / "requests.csv"
    if not path.is_file() or not requests_path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        with requests_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "complete"
        and int(summary.get("num_requests", -1)) == args.num_requests
        and int(summary.get("topk", -1)) == topk
        and summary.get("order_mode") == mode
        and list(map(int, summary.get("sample_indices") or ())) == expected_indices
        and len(rows) == args.num_requests
        and all(row.get("retrieval_idxs") for row in rows)
    )


def run_point(
    args: argparse.Namespace,
    topk: int,
    mode: str,
    expected_indices: list[int],
) -> None:
    if verified_point(args, topk, mode, expected_indices):
        print(f"SKIP_COMPLETE topk={topk} mode={mode}", flush=True)
        return
    output_dir = point_dir(args, topk, mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "numactl",
        "--physcpubind=0-63",
        "--interleave=0-3",
        str(args.python),
        str(args.benchmark),
        "--dataset",
        "nq",
        "--workflow",
        "onepass",
        "--components",
        "full",
        "--order-mode",
        mode,
        "--topk",
        str(topk),
        "--nprobe",
        "128",
        "--arrival-rate",
        str(args.arrival_rate),
        "--num-requests",
        str(args.num_requests),
        "--seed",
        str(args.arrival_seed),
        "--gpu-id",
        str(args.gpu_id),
        "--cpu-threads",
        "64",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--cpu-kv-cache-gib",
        "0",
        "--prefix-caching",
        "--direct-vllm-dispatch",
        "--cross-microbatch-pipeline",
        "--arrival-trace",
        str(args.output_root / "arrival_trace.npz"),
        "--retrieval-reference-csv",
        str(args.output_root / f"retrieval_reference_topk{topk}.csv"),
        "--require-clean-resources",
        "--max-preflight-gpu-memory-mib",
        "256",
        "--max-preflight-gpu-util-percent",
        "5",
        "--max-preflight-cpu-busy-percent",
        "20",
        "--max-preflight-cpu-max-busy-percent",
        "50",
        "--runtime-resource-sample-seconds",
        "2",
        "--max-runtime-external-cpu-busy-percent",
        "50",
        "--output-dir",
        str(output_dir),
    ]
    record = {
        "argv": command,
        "topk": topk,
        "mode": mode,
        "benchmark_sha256": file_sha256(args.benchmark),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    write_json(output_dir / "command.json", record)
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "MNEMORAG_DOC_HOTNESS_PRIOR_PATH": str(args.hotness_prior),
        "MNEMORAG_DOC_HOTNESS_FREEZE_PRIOR": "1",
        "MNEMORAG_INLINE_RELEVANCE_RANK": "0",
        "MNEMORAG_CONDITIONAL_INLINE_RANK": "0",
        "MNEMORAG_STRONG_PRIORITY_NOTE": "0",
        "MNEMORAG_CONCISE_STRONG_PRIORITY_NOTE": "0",
        "MNEMORAG_TERSE_PRIORITY_NOTE": "0",
        "MNEMORAG_COMPACT_PRIORITY_NOTE": "0",
        "MNEMORAG_OMIT_PRIORITY_NOTE": "0",
    }
    print(f"RUN topk={topk} mode={mode}", flush=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    for attempt in range(1, 21):
        stderr_offset = stderr_path.stat().st_size if stderr_path.exists() else 0
        try:
            with (
                stdout_path.open("a", encoding="utf-8") as stdout,
                stderr_path.open("a", encoding="utf-8") as stderr,
            ):
                subprocess.run(
                    command,
                    cwd=args.project_root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                )
            break
        except subprocess.CalledProcessError:
            with stderr_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(stderr_offset)
                failure = stream.read()
            if "Resource preflight failed:" not in failure or attempt == 20:
                raise
            print(
                f"RETRY_RESOURCE_PREFLIGHT topk={topk} mode={mode} "
                f"attempt={attempt}/20",
                flush=True,
            )
            time.sleep(30)
    if not verified_point(args, topk, mode, expected_indices):
        raise RuntimeError(f"Point failed validation: topk={topk} mode={mode}")


def read_requests(
    args: argparse.Namespace, topk: int, mode: str
) -> list[dict[str, str]]:
    path = summary_path(args, topk, mode).parent / "requests.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finalize(args: argparse.Namespace) -> None:
    rows_out: list[dict[str, object]] = []
    audit: dict[str, object] = {"topks": {}}
    for topk in TOPKS:
        rows = {mode: read_requests(args, topk, mode) for mode in MODES}
        for mode in MODES:
            if len(rows[mode]) != args.num_requests:
                raise RuntimeError(f"Incomplete rows at top-k={topk}, mode={mode}")
        changed = 0
        hint_same_as_hotness = 0
        candidate_sets_equal = 0
        for index in range(args.num_requests):
            ids = {
                mode: list(map(int, json.loads(rows[mode][index]["retrieval_idxs"])))
                for mode in MODES
            }
            changed += int(ids["original"] != ids["hotness"])
            hint_same_as_hotness += int(ids["hotness"] == ids["hotness_hint"])
            candidate_sets_equal += int(
                set(ids["original"]) == set(ids["hotness"]) == set(ids["hotness_hint"])
            )
        if (changed, hint_same_as_hotness, candidate_sets_equal) != (
            args.num_requests,
            args.num_requests,
            args.num_requests,
        ):
            raise RuntimeError(
                f"Reordering audit failed at top-k={topk}: "
                f"changed={changed}, hint_matches={hint_same_as_hotness}, "
                f"candidate_sets={candidate_sets_equal}"
            )
        audit["topks"][str(topk)] = {
            "original_to_hotness_changed": changed,
            "hotness_hint_order_matches_hotness": hint_same_as_hotness,
            "candidate_sets_equal": candidate_sets_equal,
        }
        for metric in ("f1", "acc"):
            output = {"topk": topk, "n": args.num_requests, "metric": metric.upper()}
            for mode in MODES:
                values = np.asarray(
                    [float(row[metric]) for row in rows[mode]],
                    dtype=np.float64,
                )
                output[mode] = float(values.mean())
            rows_out.append(output)

    final_dir = args.output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    csv_path = final_dir / "table3_scheme1_reordered512.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("topk", "n", "metric", *MODES),
        )
        writer.writeheader()
        writer.writerows(rows_out)
    write_json(final_dir / "table3_scheme1_reordered512.json", rows_out)
    write_json(final_dir / "reordering_audit.json", audit)

    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Effect of document ordering strategies on 512 queries "
        "that are reordered at every evaluated top-$k$.}",
        "\\label{tab:order}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "Top-$k$ & Metric & Original & Hotness & Hotness+Hint \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows_out):
        prefix = f"{row['topk']}" if row["metric"] == "F1" else ""
        tex.append(
            f"{prefix} & {row['metric']} & "
            f"{row['original']:.4f} & {row['hotness']:.4f} & "
            f"{row['hotness_hint']:.4f} \\\\"
        )
        if row["metric"] == "ACC" and index != len(rows_out) - 1:
            tex.append("\\midrule")
    tex.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}"))
    (final_dir / "table3_scheme1_reordered512.tex").write_text(
        "\n".join(tex) + "\n",
        encoding="utf-8",
    )
    write_json(
        args.output_root / "campaign_complete.json",
        {
            "status": "complete",
            "points": len(TOPKS) * len(MODES),
            "requests_per_point": args.num_requests,
            "table_csv": str(csv_path.resolve()),
            "table_csv_sha256": file_sha256(csv_path),
            "reordering_audit": audit,
        },
    )
    print(f"CAMPAIGN_COMPLETE {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation/document_ordering",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("MNEMORAG_PYTHON", sys.executable)),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/benchmark_point.py",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    )
    parser.add_argument(
        "--retrieval-ids",
        type=Path,
        default=PROJECT_ROOT
        / "results/background_motivation/cache/nq_nprobe128_top10_doc_ids.npy",
    )
    parser.add_argument(
        "--retrieval-scores",
        type=Path,
        default=PROJECT_ROOT
        / "results/background_motivation/cache/nq_nprobe128_top10_scores.npy",
    )
    parser.add_argument(
        "--history-indices",
        type=Path,
        default=PROJECT_ROOT
        / "results/evaluation/end_to_end/history/nq/history_indices.npy",
    )
    parser.add_argument(
        "--hotness-prior",
        type=Path,
        default=PROJECT_ROOT
        / "results/evaluation/end_to_end/history/nq/document_hotness_prior.json",
    )
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=314159)
    parser.add_argument("--arrival-seed", type=int, default=271828)
    parser.add_argument("--arrival-rate", type=float, default=8.0)
    parser.add_argument("--gpu-id", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    manifest = prepare_workload(args)
    expected_indices = list(
        map(
            int,
            np.load(manifest["selected_indices_path"], allow_pickle=False),
        )
    )
    for topk in TOPKS:
        for mode in MODES:
            run_point(args, topk, mode, expected_indices)
    finalize(args)


if __name__ == "__main__":
    main()
