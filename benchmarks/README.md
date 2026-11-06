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
| `build_next_use_profile.py` | Build paired GPU next-use and productive CPU-admission profiles from an audited block trace |
| `run_full_nq_heat_contract.py` | Fail-closed 79,168-request NQ comparison of the current and legacy heat formulas under the Figure 14 top-k=3 contract |
| `run_mnemo_ablation.py` | Similarity-aware retrieval, Hotness, Async, and Full component ablation |
| `workload_identity.py` | Fail-closed role, sample, warmup, and trace identity audit |
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

Before an ad hoc formal-QPS run, bind the intended role explicitly. This
prevents another 512-row artifact, such as the document-ordering stress set,
from being accepted merely because its request count matches:

```bash
python -m benchmarks.workload_identity \
  --selection-manifest /path/to/nq/selection_manifest.json \
  --expected-role formal_qps_proxy \
  --arrival-trace /path/to/workload.npz \
  --expected-dataset nq \
  --expected-request-count 512
```

The full campaign runner performs this audit automatically, embeds the
canonical identity in the arrival-trace manifest, and rechecks it during
comparison validation.

The full-NQ heat runner additionally requires FP8 GPU KV at 0.90 utilization,
64 sequences, 130,000 CPU KV blocks, active H2D and D2H transfers, at least
99% next-use profile matching, and nonzero selected-formula/tie-break counters.
CPU KV capacity is derived from the effective KV dtype (one byte per FP8
element), not the model-weight dtype. `--cpu-kv-pinned-tier-gib` and
`--cpu-kv-lazy-mmap-dir` preserve the exact block count while allowing the
remainder of the pool to use audited tmpfs or disk-backed mmap storage; the
runner fails before launch when a disk-backed directory lacks sufficient free
space.
Figure 14 reports one-second samples of the token-weighted hit rate over the
most recent 1,000 requests, not the final cumulative rate or the maximum of
per-request ratios. The full-NQ runner therefore checks the historical top-k=3
periodic mean (`0.225027`) and maximum (`0.281`) within a 3-percentage-point
hardware/revalidation band. The historical cumulative-series maximum
(`0.303006`) is retained as provenance, but is not compared to a different
metric. Both formulas must also agree within 0.5 percentage points. A
profile-only run is an explicit ablation and cannot pass this contract.
