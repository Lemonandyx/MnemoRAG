#!/usr/bin/env python3
"""Prepare immutable artifacts used by the retrieval-only baseline campaign.

The source Wikipedia embeddings are the same normalized E5-large-v2 vectors
used to build MnemoRAG's IVF4096,Flat index.  The generated artifacts are:

* a representative DiskANN-format float32 sample used only to train QVCache's
  regional PCA model; and
* HaS's native IVF-PQ fuzzy-channel index.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORS = Path(
    os.environ.get(
        "MNEMORAG_VECTOR_PATH",
        PROJECT_ROOT / "assets/wiki18_2100w/vectors.f16.npy",
    )
).resolve()
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "MNEMORAG_BASELINE_ARTIFACTS",
        PROJECT_ROOT / "results/retrieval_baseline_artifacts",
    )
).resolve()


def prepare_qvcache_sample(
    vectors: np.ndarray, output: Path, rows: int, seed: int
) -> None:
    if output.exists() and output.stat().st_size == 8 + rows * vectors.shape[1] * 4:
        print(f"QV_SAMPLE_EXISTS {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(vectors), rows, replace=False))
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<II", rows, vectors.shape[1]))
        for start in range(0, rows, 4096):
            batch = np.asarray(
                vectors[selected[start : start + 4096]], dtype=np.float32
            )
            batch.tofile(stream)
    temporary.replace(output)
    print(f"QV_SAMPLE_COMPLETE {output}", flush=True)


def prepare_has_index(
    vectors: np.ndarray,
    output: Path,
    train_rows: int,
    add_batch: int,
    seed: int,
) -> None:
    if output.exists():
        index = faiss.read_index(str(output))
        if index.ntotal == len(vectors) and index.d == vectors.shape[1]:
            print(f"HAS_INDEX_EXISTS {output}", flush=True)
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(vectors), train_rows, replace=False))
    train = np.asarray(vectors[selected], dtype=np.float32)
    faiss.normalize_L2(train)
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer,
        vectors.shape[1],
        8192,
        128,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    print(f"HAS_TRAIN_START rows={train_rows}", flush=True)
    index.train(train)
    del train
    for start in range(0, len(vectors), add_batch):
        batch = np.asarray(vectors[start : start + add_batch], dtype=np.float32)
        faiss.normalize_L2(batch)
        index.add(batch)
        print(f"HAS_ADD {index.ntotal}/{len(vectors)}", flush=True)
    index.nprobe = 64
    faiss.write_index(index, str(temporary))
    temporary.replace(output)
    print(f"HAS_INDEX_COMPLETE {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qv-sample-rows", type=int, default=256_000)
    parser.add_argument("--has-train-rows", type=int, default=1_000_000)
    parser.add_argument("--has-add-batch", type=int, default=100_000)
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[: args.cpu_threads]))
    faiss.omp_set_num_threads(args.cpu_threads)
    started = time.time()
    vectors = np.load(VECTORS, mmap_mode="r")
    if vectors.shape != (21_015_324, 1024) or vectors.dtype != np.float16:
        raise RuntimeError(
            f"Unexpected Wikipedia embedding matrix: {vectors.shape} {vectors.dtype}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_qvcache_sample(
        vectors,
        args.output_dir / "wiki18_e5_qvcache_pca_sample.f32.bin",
        args.qv_sample_rows,
        args.seed,
    )
    prepare_has_index(
        vectors,
        args.output_dir / "wiki18_e5_has_ivfpq8192_m128_8bit.index",
        args.has_train_rows,
        args.has_add_batch,
        args.seed,
    )
    manifest = {
        "source_vectors": str(VECTORS),
        "source_shape": list(vectors.shape),
        "source_dtype": str(vectors.dtype),
        "qvcache_sample_rows": args.qv_sample_rows,
        "has_index": {
            "type": "IndexIVFPQ",
            "nlist": 8192,
            "M": 128,
            "nbits": 8,
            "nprobe": 64,
            "train_rows": args.has_train_rows,
        },
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
