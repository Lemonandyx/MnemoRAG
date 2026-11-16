# Benchmarks

Entry points for the experiments in the MnemoRAG paper.

## Environment

Set the resource roots before running an experiment:

```bash
export MNEMORAG_ASSET_ROOT=/path/to/resources
export MNEMORAG_VLLM_ROOT=/path/to/patched/vllm
export HEDRARAG_ROOT=/path/to/HedraRAG
```

The expected asset root contains the embedding model, generation model,
datasets, Wikipedia corpus, and IVF index described in the paper.

## Public entry points

| Program | Purpose |
|---|---|
| `benchmark_point.py` | Run one isolated MnemoRAG or vanilla serving point |
| `run_full_e2e_campaign.py` | Run the end-to-end evaluation campaign |
| `run_retrieval_six_system_four_dataset.py` | Run the retrieval comparison |
| `run_mnemo_ablation.py` | Run the component ablation |

Examples:

```bash
python -m benchmarks.run_full_e2e_campaign --help
python -m benchmarks.run_retrieval_six_system_four_dataset --help
python -m benchmarks.run_mnemo_ablation --help
```

Supporting modules provide workload generation, evaluation utilities, and
baseline adapters.
