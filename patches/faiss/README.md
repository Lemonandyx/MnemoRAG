# Faiss integration

MnemoRAG uses Faiss 1.8.0 without a source patch. Retrieval is implemented in
[retriever.py](../../mnemorag/retriever/retriever.py).

```bash
python -m pip install faiss-cpu==1.8.0
```

For the HedraRAG GPU baseline, set `HEDRARAG_FAISS_ROOT` to the Faiss build
provided with HedraRAG.
