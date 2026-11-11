"""Build the dense Faiss index used by MnemoRAG."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import faiss
import numpy as np

from mnemorag.retriever.encoder import Encoder
from mnemorag.retriever.utils import load_corpus


class IndexBuilder:
    """Encode a corpus and construct a dense inner-product Faiss index."""

    def __init__(
        self,
        *,
        retrieval_method: str,
        model_path: str,
        corpus_path: str,
        save_dir: str,
        max_length: int = 512,
        batch_size: int = 64,
        use_fp16: bool = True,
        pooling_method: str = "mean",
        instruction: str | None = None,
        faiss_type: str = "IVF4096,Flat",
        embedding_path: str | None = None,
        save_embedding: bool = False,
        faiss_gpu: bool = False,
    ) -> None:
        self.retrieval_method = retrieval_method
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = Path(save_dir)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)
        self.pooling_method = pooling_method
        self.instruction = instruction
        self.faiss_type = faiss_type
        self.embedding_path = embedding_path
        self.save_embedding = bool(save_embedding)
        self.faiss_gpu = bool(faiss_gpu)

    @property
    def index_path(self) -> Path:
        name = f"{self.retrieval_method}_{self.faiss_type}.index"
        return self.save_dir / name

    @property
    def embedding_save_path(self) -> Path:
        return self.save_dir / f"emb_{self.retrieval_method}.memmap"

    def _encoder(self) -> Encoder:
        return Encoder(
            model_name=self.retrieval_method,
            model_path=self.model_path,
            pooling_method=self.pooling_method,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            instruction=self.instruction,
        )

    def _load_or_encode(self) -> np.ndarray:
        corpus = load_corpus(self.corpus_path)
        encoder = self._encoder()
        hidden_size = int(encoder.model.config.hidden_size)
        if self.embedding_path:
            return np.memmap(
                self.embedding_path,
                mode="r",
                dtype=np.float32,
            ).reshape(len(corpus), hidden_size)

        embeddings = encoder.encode(
            corpus["contents"],
            batch_size=self.batch_size,
            is_query=False,
        )
        if self.save_embedding:
            target = np.memmap(
                self.embedding_save_path,
                mode="w+",
                dtype=embeddings.dtype,
                shape=embeddings.shape,
            )
            target[:] = embeddings
            target.flush()
        return embeddings

    def build(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.asarray(self._load_or_encode(), dtype=np.float32)
        index = faiss.index_factory(
            embeddings.shape[1],
            self.faiss_type,
            faiss.METRIC_INNER_PRODUCT,
        )
        if self.faiss_gpu:
            clone_options = faiss.GpuMultipleClonerOptions()
            clone_options.useFloat16 = True
            clone_options.shard = True
            index = faiss.index_cpu_to_all_gpus(index, co=clone_options)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
        if self.faiss_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, str(self.index_path))
        return self.index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_method", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--pooling_method", default="mean")
    parser.add_argument("--instruction")
    parser.add_argument("--faiss_type", default="IVF4096,Flat")
    parser.add_argument("--embedding_path")
    parser.add_argument("--save_embedding", action="store_true")
    parser.add_argument("--faiss_gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    index_path = IndexBuilder(**vars(parse_args())).build()
    print(index_path)


if __name__ == "__main__":
    main()
