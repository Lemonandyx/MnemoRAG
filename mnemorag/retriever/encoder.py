from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

from mnemorag.retriever.utils import load_model, parse_query, pooling
from mnemorag.utils import get_device


class Encoder:
    """Encode queries and documents as dense vectors."""

    def __init__(
        self,
        model_name,
        model_path,
        pooling_method,
        max_length,
        use_fp16=True,
        instruction=None,
        silent=False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.silent = silent
        self.model, self.tokenizer = load_model(
            model_path=model_path, use_fp16=use_fp16
        )

    @torch.inference_mode()
    def single_batch_encode(
        self,
        query_list: Union[List[str], str],
        is_query=True,
        return_torch=False,
    ) -> Union[np.ndarray, torch.Tensor]:
        query_list = parse_query(
            self.model_name, query_list, self.instruction, is_query
        )
        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = inputs.to(get_device(), non_blocking=True)

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(
            output.get("pooler_output"),
            output.get("last_hidden_state"),
            inputs["attention_mask"],
            self.pooling_method,
        )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        if return_torch:
            return query_emb.detach()
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

    @torch.inference_mode()
    def encode(
        self,
        query_list: List[str],
        batch_size=64,
        is_query=True,
        return_torch=False,
    ) -> Union[np.ndarray, torch.Tensor]:
        query_emb = []
        for i in tqdm(
            range(0, len(query_list), batch_size),
            desc="Encoding process: ",
            disable=self.silent,
        ):
            query_emb.append(
                self.single_batch_encode(
                    query_list[i : i + batch_size],
                    is_query,
                    return_torch=return_torch,
                )
            )
        if return_torch:
            return torch.cat(query_emb, dim=0)
        return np.concatenate(query_emb, axis=0)
