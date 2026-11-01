# Benchmarks

This directory contains the stable experiment interfaces used by the latest
MnemoRAG manuscript. Historical campaign copies, failed runs, date-suffixed
backups, raw model outputs, and machine-specific launch wrappers are excluded.

## Shared evaluation contract

- 64 physical CPU cores per point, configured as 64 retrieval workers with one
  Faiss OMP thread each.
- One selected GPU per point unless model scaling explicitly requests more.
- E5-large-v2, a 21M-passage Wikipedia corpus, and one shared IVF4096-Flat
  index across comparable retrieval systems.
- Llama-3-8B-Instruct with aligned prompts, context limits, output limits, and
  GPU-memory fractions.
- Poisson arrivals replayed from one recorded trace for every system in a
  comparison group.
- A fresh process, Query Base, and vLLM prefix cache for every experimental
  point.
- NQ, HotpotQA, SQuAD, and 2WikiMultiHopQA under One-pass, RECOMP, and
  three-round IRG where applicable.

Expected resources:

```text
$MNEMORAG_ASSET_ROOT/
  models/e5-large-v2/
  models/llama3-8B-instruct/
  qa/datasets/{nq,hotpotqa,squad,2wikimultihopqa}/
  qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index
  qa/wiki18_2100w/corpus/wiki18_2100w.jsonl
```

Optional baseline roots are supplied through `HEDRARAG_ROOT`,
`HEDRARAG_FAISS_ROOT`, `GPTCACHE_PYTHON`, `QVCACHE_ROOT`, `HAS_ROOT`, and
`PROXIMIPY_LIBRARY`.

## Entry points

| Program | Purpose |
|---|---|
| `benchmark_point.py` | One isolated MnemoRAG or vanilla serving point |
| `orchestrate_full_e2e.py` | Full end-to-end campaign orchestration |
| `run_full_e2e_campaign.py` | One resumable end-to-end campaign phase |
| `hedra_flash_point.py` | FlashRAG and HedraRAG under the shared workflow contract |
| `run_retrieval_six_system_four_dataset.py` | Retrieval-only comparison over four datasets |
| `run_retrieval_cache_campaign.py` | nprobe and arrival-rate cache sweep |
| `run_prefix_campaign.py` | Prefix-cache policy comparison |
| `run_mnemo_ablation.py` | Similarity-aware retrieval, Hotness, Async, and Full component ablation |
| `document_ordering/run.py` | 512-request ordering-hint validation |
| `motivation/` | Background and motivation measurements |
| `baselines/` | Baseline-specific adapters |

Example:

```bash
export MNEMORAG_ASSET_ROOT=/path/to/resources
export MNEMORAG_VLLM_ROOT=/path/to/patched/vllm

python -m benchmarks.run_mnemo_ablation \
  --output-dir results/evaluation/component_ablation \
  --gpu-id 0 \
  --cpu-threads 64
```

Use `--help` on each program for its complete contract. Raw results belong
under `results/` and are excluded from version control.
