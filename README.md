# MnemoRAG: Exploiting Inter-Request Correlations for High-Performance RAG Serving

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/MnemoRAG.pdf)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MnemoRAG is a RAG serving system that converts correlations across requests into
lower retrieval, prefill, and scheduling overhead. It combines similarity-aware
retrieval, hotness-driven prefix caching, and asynchronous cross-request
execution while preserving each request's retrieval-before-generation
dependency.

**[Read the latest paper](paper/MnemoRAG.pdf)**

## Overview

MnemoRAG integrates three complementary mechanisms:

- **Similarity-aware retrieval** reuses historical search regions and
  adapts IVF probing to query similarity.
- **Hotness-driven prefix caching** orders retrieved documents by access
  hotness, preserves relevance through a tail ordering hint, and manages GPU
  and CPU KV blocks using hotness and recency.
- **Asynchronous RAG scheduling** admits subsequent retrieval work without
  waiting for ongoing generation and overlaps CPU search, GPU query matching,
  KV restoration, prefill, and decoding.

The artifact contains the current runtime, a sparse vLLM overlay, controlled
benchmark entry points, paper plotting programs, verification tests, and the
latest paper. Models, corpora, indexes, request traces, and experimental result
tables are intentionally excluded.

## Repository layout

| Path | Description |
|---|---|
| [`mnemorag/`](mnemorag/) | Retrieval, generation, caching, scheduling, and public serving interfaces |
| [`benchmarks/`](benchmarks/) | End-to-end, retrieval, prefix-cache, ablation, baseline, and motivation workloads |
| [`configs/`](configs/) | Paper configuration with machine-independent resource placeholders |
| [`figures/`](figures/) | Plotting programs organized by experimental claim |
| [`patches/vllm/`](patches/vllm/) | Sparse overlay for the tested vLLM 0.11.2 development tree |
| [`paper/`](paper/) | Latest manuscript |
| [`examples/`](examples/) | Minimal end-to-end smoke test |
| [`scripts/`](scripts/) | Index preparation, corpus preprocessing, and overlay utilities |
| [`tests/`](tests/) | Unit, scheduler, campaign-contract, and cache/offload tests |

## Requirements

- Linux x86-64
- Python 3.10--3.12
- CUDA-compatible NVIDIA GPU
- Faiss 1.8.0
- vLLM 0.11.2 development base

The paper evaluation uses E5-large-v2, Llama-3-8B-Instruct, the 21M-passage
Wikipedia corpus, and a shared IVF4096-Flat index. CPU-only checks are sufficient
for most unit tests but not for full serving reproduction.

## Installation

```bash
git clone git@github.com:Lemonandyx/MnemoRAG.git
cd MnemoRAG

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Apply the sparse overlay to a clean compatible vLLM checkout and install that
checkout separately:

```bash
scripts/apply_vllm_patch.sh /path/to/vllm
python -m pip install -e /path/to/vllm
```

The overlay adds document-heat metadata, KV accounting, hotness-aware CPU
offload, and asynchronous CPU/GPU transfer observability. The script validates
the target before copying any files.

## Data and configuration

Copy [`configs/paper_a100.json`](configs/paper_a100.json) and replace the
resource root with local paths. Alternatively, expose the shared assets through
environment variables:

```bash
export MNEMORAG_ASSET_ROOT=/path/to/resources
export MNEMORAG_VLLM_ROOT=/path/to/patched/vllm
export HEDRARAG_ROOT=/path/to/HedraRAG          # baseline comparisons only
export GPTCACHE_PYTHON=/path/to/gptcache/python # baseline comparisons only
```

The expected asset layout and the cross-system evaluation contract are
documented in [`benchmarks/README.md`](benchmarks/README.md). No dataset, model,
corpus, or index is downloaded automatically.

## Quick start

```python
import asyncio

from mnemorag import Config, Dataset, MnemoRAGPipeline

config = Config(config_file_path="configs/local_nq.yaml")
dataset = Dataset(config=config, dataset_path="/path/to/nq/test.jsonl")
pipeline = MnemoRAGPipeline(config)
result = asyncio.run(pipeline.run_stream(dataset))
result.save("output/nq_predictions.json")
pipeline.close()
```

For a minimal integration check with the paper assets:

```bash
python examples/smoke_test.py \
  --asset-root "$MNEMORAG_ASSET_ROOT" \
  --gpu-id 0 \
  --cpu-threads 64 \
  --num-requests 2
```

## Reproducing experiments

The stable entry points are grouped under [`benchmarks/`](benchmarks/):

```bash
# Four datasets, three workflows, and increasing request rates
python -m benchmarks.orchestrate_full_e2e --help

# Retrieval-only comparison under shared IVF4096-Flat assets
python -m benchmarks.run_retrieval_six_system_four_dataset --help

# Similarity-aware retrieval, Hotness, Async, and Full component ablation
python -m benchmarks.run_mnemo_ablation --help

# Ordering-hint validation on 512 always-reordered NQ requests
python -m benchmarks.document_ordering.run --help
```

Every formal point starts a fresh serving process. Workload manifests record
request identities, arrival traces, source hashes, runtime contracts, and
resource checks. Raw outputs belong under `results/`, which is ignored by Git.

## Reproducing figures

Plotting programs are organized by claim under [`figures/`](figures/). The
aggregate figures accept their input tables explicitly:

```bash
python figures/end_to_end/plot.py --input /path/to/end_to_end.csv
python figures/component_ablation/plot.py --input /path/to/component_ablation.csv
```

The programs emit PNG and PDF files under `output/`. Experimental tables are
not distributed with the source release.

## Verification

```bash
python -m compileall -q mnemorag benchmarks figures patches/vllm
pytest -q
```

GPU integration tests additionally require the configured models, index,
corpus, and CUDA runtime.

## Upstream projects and license

MnemoRAG retains selected interfaces from
[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) and applies a sparse overlay
to [vLLM](https://github.com/vllm-project/vllm). Attribution details are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). MnemoRAG is released under
the [MIT License](LICENSE).
