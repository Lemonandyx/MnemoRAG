import math
import re
import warnings
from typing import Any, Dict, List, Union

import datasets
import numpy as np
from transformers import AutoModel, AutoTokenizer

from mnemorag.utils import get_device

_has_printed_instruction = False


def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert numpy objects in nested dictionaries or lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj


def judge_zh(input_str: str):
    assert isinstance(input_str, str), input_str
    if len(input_str) == 0:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", input_str))


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(get_device())
    print("Retriever Model loaded to device:", get_device())
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )

    return model, tokenizer


def pooling(
    pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"
):
    if last_hidden_state is None and pooling_method in ["mean", "cls"]:
        warnings.warn("last_hidden_state is None, using pooler_output instead.")
        pooling_method = "pooler"

    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def set_default_instruction(model_name, is_query=True, is_zh=False):
    instruction = ""
    if "e5" in model_name.lower():
        if is_query:
            instruction = "query: "
        else:
            instruction = "passage: "

    if "bge" in model_name.lower():
        if is_query:
            if "zh" in model_name.lower() or is_zh:
                instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                instruction = (
                    "Represent this sentence for searching relevant passages: "
                )

    return instruction


def parse_query(model_name, query_list, instruction=None, is_query=True):
    """Apply the retrieval instruction expected by the embedding model."""
    global _has_printed_instruction

    if isinstance(query_list, str):
        query_list = [query_list]

    if instruction is not None:
        instruction = instruction.strip() + " "
    else:
        instruction = set_default_instruction(
            model_name, is_query=is_query, is_zh=judge_zh(query_list[0])
        )
    if not _has_printed_instruction:
        if instruction == "":
            warnings.warn("Instruction is not set")
        else:
            print(f"Use `{instruction}` as retreival instruction")
        _has_printed_instruction = True

    query_list = [instruction + query for query in query_list]

    return query_list


def load_corpus(corpus_path: str):
    if corpus_path.endswith(".jsonl"):
        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    elif corpus_path.endswith(".parquet"):
        corpus = datasets.load_dataset("parquet", data_files=corpus_path, split="train")
        corpus = corpus.cast_column("image", datasets.Image())
    else:
        raise NotImplementedError("Corpus format not supported!")
    if "contents" not in corpus.features:
        try:
            print("No `contents` field found in corpus, using `text` instead.")
            corpus = corpus.map(lambda x: {"contents": x["text"]})
        except (KeyError, TypeError):
            warnings.warn("Corpus contains neither a `contents` nor a `text` field.")
    return corpus


def load_docs(corpus, doc_idxs: List[int]):
    indices = [int(idx) for idx in doc_idxs]
    if not indices:
        return []
    corpus_size = len(corpus)
    minimum = min(indices)
    maximum = max(indices)
    if minimum < 0 or maximum >= corpus_size:
        raise IndexError(
            "Document ids must be within corpus bounds: "
            f"min={minimum}, max={maximum}, corpus_size={corpus_size}"
        )
    if isinstance(corpus, datasets.Dataset):
        batch = corpus[indices]
        keys = list(batch)
        columns = [batch[key] for key in keys]
        return [dict(zip(keys, values)) for values in zip(*columns)]
    return [corpus[idx] for idx in indices]


def align_probe_count(n: int) -> int:
    """Round a probe count up to the nearest power of two."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def adaptive_probe_budget(
    similarity: float,
    similarity_low: float,
    similarity_high: float,
    minimum: int,
    maximum: int,
    gamma: float = 2.0,
    rounding: str = "ceil",
    curve: str = "power_decay",
) -> int:
    """Map similarity to a bounded IVF probe budget."""
    minimum = int(minimum)
    maximum = int(maximum)
    gamma = float(gamma)
    if minimum < 1 or maximum < minimum:
        raise ValueError("Probe bounds must satisfy 1 <= minimum <= maximum")
    if similarity_high <= similarity_low:
        raise ValueError("Similarity normalization requires high > low")
    if gamma <= 0:
        raise ValueError("similarity_nprobe_gamma must be positive")

    normalized = (float(similarity) - similarity_low) / (
        similarity_high - similarity_low
    )
    normalized = min(1.0, max(0.0, normalized))
    if curve == "power_decay":
        budget_fraction = math.pow(1.0 - normalized, gamma)
    elif curve == "late_decay":
        budget_fraction = 1.0 - math.pow(normalized, gamma)
    else:
        raise ValueError("similarity_nprobe_curve must be power_decay or late_decay")
    continuous = minimum + (maximum - minimum) * budget_fraction
    if rounding == "ceil":
        budget = math.ceil(continuous - 1e-12)
    elif rounding == "nearest":
        budget = math.floor(continuous + 0.5)
    elif rounding == "floor":
        budget = math.floor(continuous + 1e-12)
    elif rounding == "power_of_two":
        budget = align_probe_count(math.floor(continuous + 1e-12))
    else:
        raise ValueError(
            "similarity_nprobe_rounding must be one of: "
            "ceil, nearest, floor, power_of_two"
        )
    return min(maximum, max(minimum, int(budget)))
