# MnemoRAG: Exploiting Inter-Request Correlations for High-Performance RAG Serving

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/MnemoRAG.pdf)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MnemoRAG is a RAG serving system that converts correlations across requests into
lower retrieval, prefill, and scheduling overhead. It combines similarity-aware
retrieval, hotness-driven prefix caching, and asynchronous RAG scheduling.

**[Read the latest paper](paper/MnemoRAG.pdf)**

## Overview

MnemoRAG integrates three complementary mechanisms:

- **Similarity-aware retrieval** reuses historical search regions and
  adapts IVF probing to query similarity.
- **Hotness-driven prefix caching** orders retrieved documents by access
  hotness, preserves relevance through a tail ordering hint, and manages GPU
  and CPU KV blocks using hotness and recency.
- **Asynchronous RAG scheduling** releases the next retrieval microbatch after
  every request in the current microbatch is admitted to prefill, then overlaps
  CPU search, GPU query matching, KV restoration, prefill, and decoding with
  one bounded look-ahead microbatch.

## Repository layout

| Path | Description |
|---|---|
| [`mnemorag/`](mnemorag/) | RAG serving implementation |
| [`benchmarks/`](benchmarks/) | Experiment scripts |
| [`configs/`](configs/) | Experiment settings |
| [`figures/`](figures/) | Plotting scripts |
| [`patches/`](patches/) | Runtime dependencies and vLLM modifications |
| [`paper/`](paper/) | Latest manuscript |
| [`examples/`](examples/) | Usage example |
| [`scripts/`](scripts/) | Setup utilities |
| [`tests/`](tests/) | Tests |

## Requirements

- Linux x86-64
- Python 3.10--3.12
- CUDA-compatible NVIDIA GPU
- Faiss 1.8.0
- vLLM 0.11.2 development base

The paper evaluation uses E5-large-v2, Llama-3-8B-Instruct, the 21M-passage
Wikipedia corpus, and an IVF4096-Flat index.

## Installation

```bash
git clone git@github.com:Lemonandyx/MnemoRAG.git
cd MnemoRAG

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Apply the vLLM modifications and install the modified checkout:

```bash
scripts/apply_vllm_patch.sh /path/to/vllm
python -m pip install -e /path/to/vllm
```

See [`patches/faiss/README.md`](patches/faiss/README.md) for Faiss setup.

## Data and configuration

Set the resource paths:

```bash
export MNEMORAG_ASSET_ROOT=/path/to/resources
export MNEMORAG_VLLM_ROOT=/path/to/patched/vllm
export HEDRARAG_ROOT=/path/to/HedraRAG          # baseline comparisons only
export GPTCACHE_PYTHON=/path/to/gptcache/python # baseline comparisons only
```

Experiment settings are in [`configs/paper_a100.json`](configs/paper_a100.json).
See [`benchmarks/README.md`](benchmarks/README.md) for the asset layout.

For index-building options, run `python -m mnemorag.retriever.index_builder --help`.

## Quick start

Run a short NQ example with the configured assets:

```bash
python examples/test.py \
  --asset-root "$MNEMORAG_ASSET_ROOT" \
  --gpu-id 0 \
  --cpu-threads 64 \
  --num-requests 2
```

## Reproducing experiments

The experiment scripts cover end-to-end serving across datasets and RAG
workflows, retrieval comparisons, and component ablations. Configure the
resource paths above, then use `--help` to view the workload and runtime
options for each entry point.

```bash
# Four datasets, three workflows, and increasing request rates
python -m benchmarks.run_full_e2e_campaign --help

# Retrieval-only comparison under shared IVF4096-Flat assets
python -m benchmarks.run_retrieval_six_system_four_dataset --help

# NQ at 8 QPS: Vanilla, w/o SAR, w/o Hotness, w/o Async, and Full
python -m benchmarks.run_mnemo_ablation --help
```

## Verification

Check Python syntax and run the test suite in the environment with the patched
vLLM installation. The tests cover retrieval reuse and guidance, document
hotness, KV-cache eviction, CPU offload, and asynchronous request scheduling.

```bash
python -m compileall -q mnemorag benchmarks figures patches/vllm
pytest -q
```

## Upstream projects and license

MnemoRAG builds on [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) and
[vLLM](https://github.com/vllm-project/vllm). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
MnemoRAG is released under the [MIT License](LICENSE).
