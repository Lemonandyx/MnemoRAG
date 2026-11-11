# MnemoRAG asynchronous execution contract

The serving path preserves retrieval-before-generation within each request and
removes that barrier across requests.

## Request-level scheduling

- `examples/quick_start/utils.py` releases requests on an open-loop arrival
  timeline. Workers drain only requests already available in the queue.
- `AsyncPipeline._retrieve_async` is the single owner of Query Base state and
  serializes/coalesces ready retrieval calls without a timed batching window.
- One-Pass submits every retrieval-ready microbatch to persistent `AsyncLLM`.
  Retrieval for microbatch *i+1* is released only after every request in
  microbatch *i* has emitted the engine-side `SCHEDULED` prefill-admission
  event.
- RECOMP applies the same admission gate to compression generation, then
  pipelines retrieval of chunk *i+1* with compression/answer generation for
  chunk *i*. Dependencies inside one request remain unchanged.
- IRG applies the admission gate within each round, overlaps generation for a
  retrieval-complete batch with the next batch, and retains the round barrier
  needed to form the next request state.
- `async_generation_queue_batches` bounds outstanding generation
  microbatches. A full queue backpressures the serialized retrieval path; the
  paper configuration uses a capacity of two, representing the active batch
  and one look-ahead batch.

## Retrieval and KV-transfer overlap

- Query encoding runs on GPU. Historical-query matching and CPU IVF coarse
  search are submitted concurrently before fine search.
- CPU-resident KV uses pinned host tensors and dedicated H2D/D2H CUDA streams.
- CUDA events track transfer completion. Requests remain in
  `WAITING_FOR_REMOTE_KVS` until H2D completes.
- The frontend prefill-admission event is derived from EngineCore's
  `SCHEDULED` event after KV readiness, not from frontend request enqueue.

## Runtime source of truth

- `patches/vllm` is the auditable sparse overlay for the tested vLLM tree.
- `scripts/apply_vllm_patch.sh` validates and applies the overlay to a clean
  compatible checkout; `MNEMORAG_VLLM_ROOT` selects that executable tree.
- Runtime and scheduler tests verify the prefill-admission event, independent
  transfer streams, KV readiness barrier, and paper priority formula.

Run the complete verification suite with:

```bash
PYTHONPATH="$MNEMORAG_VLLM_ROOT:$PWD" \
  python -m pytest -q
```
