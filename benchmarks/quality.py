"""Exact-neighbor retrieval quality, evaluated outside the serving interval."""

from __future__ import annotations

import numpy as np


def recall_at_k(candidates, reference, k: int) -> list[float]:
    """Return one set-overlap score per query with a fixed denominator k."""
    candidates = np.asarray(candidates, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if k < 1 or candidates.ndim != 2 or reference.shape != candidates.shape:
        raise ValueError("candidate and reference rows must have matching shapes")
    if candidates.shape[1] != k or np.any(candidates < 0) or np.any(reference < 0):
        raise ValueError("each result must contain k valid document IDs")
    if any(len(set(row)) != k for row in reference):
        raise ValueError("exact reference rows must contain k distinct neighbors")
    return [
        len(set(row) & set(target)) / k
        for row, target in zip(candidates.tolist(), reference.tolist())
    ]


def exact_retrieval_recall(encoder, index, queries, candidates, k: int):
    """Exhaust all IVF lists using the same encoder, index, and final queries."""
    import faiss

    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    if not isinstance(ivf, faiss.IndexIVFFlat):
        raise ValueError("exact recall requires the paper's IVF-Flat index")
    embeddings = np.asarray(
        encoder.encode(list(queries), batch_size=64, is_query=True),
        dtype=np.float32,
        order="C",
    )
    params = faiss.SearchParametersIVF()
    params.nprobe = ivf.nlist
    params.max_codes = 0
    _, reference = index.search(embeddings, k, params=params)
    return recall_at_k(candidates, reference, k)
