"""MnemoRAG serving interface."""

from mnemorag.config import Config
from mnemorag.dataset import Dataset
from mnemorag.pipeline import AsyncPipeline
from mnemorag.version import __version__


class MnemoRAGPipeline(AsyncPipeline):
    """Entry point for MnemoRAG serving."""


__all__ = ["Config", "Dataset", "MnemoRAGPipeline", "__version__"]
