# Benchmarks

This directory contains the public evaluation entry points used by the latest
MnemoRAG manuscript. Raw measurements, request traces, model outputs, local
resource paths, and exploratory utilities are not part of the source release.

## Environment

Set the resource roots before running an experiment:

```bash
export MNEMORAG_ASSET_ROOT=/path/to/resources
export MNEMORAG_VLLM_ROOT=/path/to/patched/vllm
export HEDRARAG_ROOT=/path/to/HedraRAG
```

The expected asset root contains the embedding model, generation model,
datasets, Wikipedia corpus, and IVF index described in the paper. Every formal
point starts in a fresh process and writes its output below an ignored
`results/` directory.

## Public entry points

| Program | Purpose |
|---|---|
| `benchmark_point.py` | Run one isolated MnemoRAG or vanilla serving point |
| `run_full_e2e_campaign.py` | Run the end-to-end evaluation campaign |
| `run_retrieval_six_system_four_dataset.py` | Run the retrieval comparison |
| `run_prefix_campaign.py` | Run the prefix-cache comparison |
| `run_mnemo_ablation.py` | Run the component ablation |

Examples:

```bash
python -m benchmarks.run_full_e2e_campaign --help
python -m benchmarks.run_retrieval_six_system_four_dataset --help
python -m benchmarks.run_prefix_campaign --help
python -m benchmarks.run_mnemo_ablation --help
```

Supporting modules implement shared workload construction, resource checks,
runtime validation, and baseline adapters. They are imported by the public
entry points and are not separate experimental programs.
