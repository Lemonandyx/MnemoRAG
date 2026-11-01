"""Factories for the MnemoRAG runtime components."""

from __future__ import annotations

import torch


def get_device() -> str:
    """Return the active inference device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_generator(config, **params):
    """Construct the vLLM generator used by the paper workloads."""
    if config["framework"] != "vllm":
        raise ValueError("MnemoRAG requires framework='vllm'")
    from mnemorag.generator.generator import VLLMGenerator

    return VLLMGenerator(config, **params)


def get_retriever(config):
    """Construct the dense IVF retriever used by MnemoRAG."""
    if config["use_multi_retriever"]:
        raise ValueError("MnemoRAG does not use multi-retriever routing")
    if config["retrieval_method"] in {"bm25", "splade"}:
        raise ValueError("MnemoRAG requires a dense retrieval method")
    from mnemorag.retriever.retriever import DenseRetriever

    return DenseRetriever(config)


def get_reranker(config):
    """Construct an optional dense-retrieval reranker."""
    from transformers import AutoConfig

    from mnemorag.retriever.reranker import BiReranker, CrossReranker

    architecture = AutoConfig.from_pretrained(
        config["rerank_model_path"]
    ).architectures[0]
    if "forsequenceclassification" in architecture.lower():
        return CrossReranker(config)
    return BiReranker(config)
