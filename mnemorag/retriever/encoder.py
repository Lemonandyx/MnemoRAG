from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

from mnemorag.retriever.utils import load_model, parse_query, pooling
from mnemorag.utils import get_device


class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

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
        self.gpu_num = torch.cuda.device_count()
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

        if "T5" in type(self.model).__name__ or (
            isinstance(self.model, torch.nn.DataParallel)
            and "T5" in type(self.model.module).__name__
        ):
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].shape[0], 1), dtype=torch.long
            ).to(inputs["input_ids"].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)

            pooler_output = output.get("pooler_output", None)
            last_hidden_state = output.get("last_hidden_state", None)

            query_emb = pooling(
                pooler_output,
                last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )

        if "dpr" not in self.model_name:
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

    @torch.inference_mode()
    def multi_gpu_encode(
        self, query_list: Union[List[str], str], batch_size=64, is_query=True
    ) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode(query_list, batch_size, is_query)
        return query_emb
