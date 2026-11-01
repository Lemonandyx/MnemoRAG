#!/usr/bin/env python3
"""Recompute Figure 4 nearest-query similarities in exact FP32.

This intentionally reuses only the cached E5 embeddings.  It does not rerun
retrieval latency measurements or overwrite Figures 2(b) and 5.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.environ.get(
        "MNEMORAG_BACKGROUND_OUTPUT",
        PROJECT_ROOT / "results/background_motivation",
    )
).resolve()
DATASETS = ("nq", "hotpotqa", "squad", "2wikimultihopqa")


def main() -> None:
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[:64]))
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    torch.backends.cuda.matmul.allow_tf32 = False

    source_rows = {}
    source_path = OUTPUT_DIR / "figure4_semantic_similarity.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        source_rows = {row["dataset"]: row for row in csv.DictReader(stream)}

    rows = []
    for dataset in DATASETS:
        embedding_path = OUTPUT_DIR / "cache" / f"{dataset}_query_embeddings.npy"
        embeddings = np.load(embedding_path)
        vectors = torch.from_numpy(embeddings).to("cuda", dtype=torch.float32)
        maxima = np.empty(len(embeddings), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(embeddings), 2048):
                stop = min(start + 2048, len(embeddings))
                similarities = vectors[start:stop] @ vectors.T
                local = torch.arange(stop - start, device="cuda")
                global_rows = torch.arange(start, stop, device="cuda")
                similarities[local, global_rows] = -float("inf")
                maxima[start:stop] = (
                    similarities.max(dim=1).values.clamp_(-1.0, 1.0).cpu().numpy()
                )
        del vectors
        torch.cuda.empty_cache()
        np.save(
            OUTPUT_DIR / "cache" / f"{dataset}_nearest_other_similarity_fp32.npy",
            maxima,
        )
        row = dict(source_rows[dataset])
        row["mean_inter_request_similarity"] = float(maxima.mean())
        row["min_nearest_query_similarity"] = float(maxima.min())
        row["max_nearest_query_similarity"] = float(maxima.max())
        rows.append(row)
        print(
            f"SIMILARITY_DONE dataset={dataset} mean={maxima.mean():.9f} "
            f"min={maxima.min():.9f} max={maxima.max():.9f}",
            flush=True,
        )

    temporary = source_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(source_path)
    print("SIMILARITY_RECOMPUTE_OK", flush=True)


if __name__ == "__main__":
    main()
