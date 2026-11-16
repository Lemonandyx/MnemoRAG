# Benchmarks

Entry points for the experiments in the MnemoRAG paper.

## Environment

Set the resource paths as shown in the [main README](../README.md#data-and-configuration).
The scripts expect the following layout under `MNEMORAG_ASSET_ROOT`:

```text
models/e5-large-v2/
models/llama3-8B-instruct/
qa/datasets/<dataset>/test.jsonl
qa/wiki18_2100w/corpus/wiki18_2100w.jsonl
qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index
```

## Experiments

| Program | Purpose |
|---|---|
| `benchmark_point.py` | Run one MnemoRAG or vanilla serving experiment |
| `run_full_e2e_campaign.py` | Run the end-to-end evaluation |
| `run_retrieval_six_system_four_dataset.py` | Run the retrieval comparison |
| `run_mnemo_ablation.py` | Run the component ablation |

Use `python -m benchmarks.<program> --help` to view each program's options.
