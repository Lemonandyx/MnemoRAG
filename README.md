# MnemoRAG: Exploiting Inter-Request Correlations for High-Performance RAG Serving

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/MnemoRAG.pdf)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MnemoRAG exploits correlations across requests to accelerate RAG serving.

**[Read the latest paper](paper/MnemoRAG.pdf)**

## Overview

- **Similarity-aware retrieval** reuses historical search regions and
  adapts IVF probing to query similarity.
- **Hotness-driven prefix caching** improves prefix reuse across requests.
- **Asynchronous RAG scheduling** overlaps retrieval and generation.

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

```bash
# Four datasets, three workflows, and increasing request rates
python -m benchmarks.run_full_e2e_campaign --help

# Retrieval-only comparison under shared IVF4096-Flat assets
python -m benchmarks.run_retrieval_six_system_four_dataset --help

# NQ at 8 QPS: Vanilla, w/o SAR, w/o Hotness, w/o Async, and Full
python -m benchmarks.run_mnemo_ablation --help
```

## Reproducing figures

Pass experiment results to the plotting scripts:

```bash
python figures/end_to_end/plot.py --input /path/to/end_to_end.csv
python figures/component_ablation/plot.py --input /path/to/component_ablation.csv
```

The scripts export PNG and PDF files. More examples are in
[`figures/README.md`](figures/README.md).

## Verification

```bash
python -m compileall -q mnemorag benchmarks figures patches/vllm
pytest -q
```

## Upstream projects and license

MnemoRAG builds on [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) and
[vLLM](https://github.com/vllm-project/vllm). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
MnemoRAG is released under the [MIT License](LICENSE).
