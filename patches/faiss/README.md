# Faiss integration

MnemoRAG uses Faiss 1.8.0 without a C++ source patch. The retrieval logic is
implemented in [retriever.py](../../mnemorag/retriever/retriever.py):

- Query matching and coarse search run concurrently.
- Guided retrieval combines the query's selected lists with historical result lists.
- `_search_preassigned` passes explicit list IDs, centroid distances, and
  per-call `SearchParametersIVF` to the existing `search_preassigned_c` API.
- `_search_preassigned_parallel` scans disjoint list partitions and merges their
  top-k results.

The Faiss 1.8.0 high-level Python `search_preassigned` wrapper rejects
`params` and expects the index's shared `nprobe`. The adapter uses the raw
binding so concurrent searches can supply different probe counts without
changing shared index state.

These interfaces are present in the upstream 1.8.0
[`IndexIVF.h`](https://github.com/facebookresearch/faiss/blob/v1.8.0/faiss/IndexIVF.h)
and [Python wrappers](https://github.com/facebookresearch/faiss/blob/v1.8.0/faiss/python/class_wrappers.py).

Install the dependency with:

```bash
python -m pip install faiss-cpu==1.8.0
```

The retrieval and quality checks can be run without loading a corpus or model:

```bash
python -m pytest -q tests/test_retriever_guided_mapping.py \
  tests/test_similarity_nprobe_budget.py tests/test_experiment_validation.py
```

HedraRAG's GPU retrieval baseline uses the Faiss implementation distributed
with its upstream artifact. Set `HEDRARAG_FAISS_ROOT` to that build when running
the baseline. It is a separate dependency and does not replace MnemoRAG's
standard Faiss installation.
