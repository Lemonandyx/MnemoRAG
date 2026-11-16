"""Factories for MnemoRAG runtime components."""

from __future__ import annotations

import torch

__all__ = ["get_device", "get_generator", "get_retriever"]


def get_device() -> str:
    """Return the active inference device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_generator(config, **params):
    """Construct the vLLM generator."""
    if config["framework"] != "vllm":
        raise ValueError("MnemoRAG requires framework='vllm'")
    from mnemorag.generator.generator import VLLMGenerator

    return VLLMGenerator(config, **params)


def get_retriever(config):
    """Construct the dense IVF retriever."""
    if config["use_multi_retriever"]:
        raise ValueError("MnemoRAG does not use multi-retriever routing")
    if config["retrieval_method"] in {"bm25", "splade"}:
        raise ValueError("MnemoRAG requires a dense retrieval method")
    from mnemorag.retriever.retriever import DenseRetriever

    return DenseRetriever(config)
