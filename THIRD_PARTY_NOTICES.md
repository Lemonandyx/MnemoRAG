# Third-party notices

- Selected runtime components originate from FlashRAG:
  <https://github.com/RUC-NLPIR/FlashRAG>, MIT License.
- vLLM: <https://github.com/vllm-project/vllm>, Apache License 2.0. The
  `patches/vllm/` directory in this repository is a sparse source overlay and
  does not redistribute a complete vLLM checkout.

The MnemoRAG-specific changes include historical-query matching, similarity-guided IVF search, document-hotness metadata, cross-request asynchronous scheduling, CPU KV offload integration, transfer instrumentation, and the associated evaluation drivers.
