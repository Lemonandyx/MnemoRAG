import os
import time
import warnings

import tiktoken
from transformers import AutoConfig, AutoTokenizer


class PromptTemplate:
    placeholders = ["reference", "question"]
    base_system_prompt = (
        "Answer the question based on the given document."
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}"
    )
    base_user_prompt = "Question: {question}"

    def __init__(
        self,
        config,
        system_prompt="",
        user_prompt="",
        reference_template=None,
        enable_chat=True,
    ):

        self.config = config
        self.is_openai = config["framework"] == "openai"
        self.max_input_len = config["generator_max_input_len"]
        if not self.is_openai:
            self.generator_path = config["generator_model_path"]
            model_config = AutoConfig.from_pretrained(
                self.generator_path, trust_remote_code=True
            )
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if (
                "chat" in model_name
                or "instruct" in model_name
                or config["is_reasoning"]
            ):
                self.is_chat = True
        else:
            self.is_chat = True
            self.enable_chat = True

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
        self.tokenizer = None
        self.omit_priority_note = self._env_enabled("MNEMORAG_OMIT_PRIORITY_NOTE")
        self.inline_relevance_rank = self._env_enabled("MNEMORAG_INLINE_RELEVANCE_RANK")
        self.conditional_inline_rank = self._env_enabled(
            "MNEMORAG_CONDITIONAL_INLINE_RANK"
        )
        self.strong_priority_note = self._env_enabled("MNEMORAG_STRONG_PRIORITY_NOTE")
        self.concise_strong_priority_note = self._env_enabled(
            "MNEMORAG_CONCISE_STRONG_PRIORITY_NOTE"
        )
        self.terse_priority_note = self._env_enabled("MNEMORAG_TERSE_PRIORITY_NOTE")
        self.compact_priority_note = self._env_enabled("MNEMORAG_COMPACT_PRIORITY_NOTE")
        self.stable_doc_format = self._env_enabled("MNEMORAG_STABLE_DOC_FORMAT")
        self.compact_doc_format = self._env_enabled("MNEMORAG_COMPACT_DOC_FORMAT")

    @staticmethod
    def _env_enabled(name):
        return os.environ.get(name, "0") == "1"

    def bind_tokenizer(self, tokenizer):
        """Reuse a generator-owned tokenizer when the pipeline provides one."""
        self.tokenizer = tokenizer

    def _get_tokenizer(self):
        if self.tokenizer is None:
            if self.is_openai:
                try:
                    self.tokenizer = tiktoken.encoding_for_model(
                        self.config["generator_model"]
                    )
                except Exception as e:
                    print("Error: ", e)
                    warnings.warn(
                        "This model is not supported by tiktoken. Use gpt-3.5-turbo instead."
                    )
                    self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.generator_path, trust_remote_code=True
                )
        return self.tokenizer

    def truncate_prompt(self, prompt):
        if self.is_openai:
            if self.enable_chat:
                truncated_messages = []
                total_tokens = 0
                assert isinstance(prompt, list)
                for message in prompt:
                    role_content = message["content"]
                    encoded_message = self._get_tokenizer().encode(role_content)

                    if total_tokens + len(encoded_message) <= self.max_input_len:
                        truncated_messages.append(message)
                        total_tokens += len(encoded_message)
                    else:
                        print(
                            f"The input text length is greater than the maximum length ({total_tokens + len(encoded_message)} > {self.max_input_len}) and has been truncated!"
                        )
                        remaining_tokens = self.max_input_len - total_tokens
                        truncated_message = self._get_tokenizer().decode(
                            encoded_message[:remaining_tokens]
                        )
                        message["content"] = truncated_message
                        truncated_messages.append(message)
                        break
            else:
                assert isinstance(prompt, str)
                tokenized_prompt = self._get_tokenizer().encode(
                    prompt, allowed_special={"<|endoftext|>"}
                )
                half = int(self.max_input_len / 2)
                truncated_messages = self._get_tokenizer().decode(
                    tokenized_prompt[:half]
                ) + self._get_tokenizer().decode(tokenized_prompt[-half:])

            return truncated_messages

        else:
            assert isinstance(prompt, str)
            tokenized_prompt_st_time = time.time()
            try:
                tokenized_prompt = (
                    self._get_tokenizer()
                    .encode(prompt, truncation=False, return_tensors="pt")
                    .input_ids[0]
                )
            except (AttributeError, TypeError):
                tokenized_prompt = self._get_tokenizer().encode(
                    prompt, truncation=False, return_tensors="pt"
                )[0]
            tokenized_prompt_ed_time = time.time()
            print(
                f"[MnemoRAG] tokenizer.encode time: {tokenized_prompt_ed_time - tokenized_prompt_st_time:.4f} seconds"
            )
            if len(tokenized_prompt) >= self.max_input_len:
                print(
                    f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!"
                )
                half = int(self.max_input_len / 2) - 20
                prompt = self._get_tokenizer().decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + self._get_tokenizer().decode(
                    tokenized_prompt[-half:], skip_special_tokens=True
                )
            return prompt

    def get_string(
        self,
        question=None,
        retrieval_result=None,
        original_retrieval_result=None,
        formatted_reference=None,
        previous_gen=None,
        messages=None,
        return_document_spans=False,
        **params,
    ):
        reference_parts = []
        doc_id_to_doc_number = {}
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    return self.truncate_prompt(messages)
                else:
                    prompt = self._get_tokenizer().apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message["content"] for message in messages if message["content"]]
                )
                return self.truncate_prompt(prompt)

        if formatted_reference is None:
            if retrieval_result is not None:
                (
                    formatted_reference,
                    doc_id_to_doc_number,
                    reference_parts,
                ) = self.format_reference(
                    retrieval_result,
                    original_retrieval_result=original_retrieval_result,
                    return_parts=True,
                )
            else:
                formatted_reference = ""

        system_prompt = self.system_prompt
        if previous_gen is not None and previous_gen.strip() != "":
            system_prompt += (
                "\n\nKnown information from previous iteration:\n"
                f"{previous_gen}\n"
                "Note: Do not repeat the draft answer verbatim unless no improvement is possible.\n"
            )
        # system_prompt = self.base_system_prompt
        if original_retrieval_result is not None:
            originalOrder = self.original_doc_order(
                original_retrieval_result, doc_id_to_doc_number
            )
        else:
            originalOrder = ""
        if self.inline_relevance_rank and (
            not self.conditional_inline_rank or originalOrder
        ):
            system_prompt += (
                "\nLower relevance ranks are more important (1 is highest)."
            )
        elif self.strong_priority_note and originalOrder:
            natural_order = originalOrder.replace(" → ", ", then ")
            if self.concise_strong_priority_note:
                system_prompt += f"\nRelevance (highest first): {natural_order}."
            else:
                system_prompt += (
                    "\nIMPORTANT: The document display order reflects cache heat, "
                    "not relevance. For answering, evidence relevance is "
                    f"{natural_order}, from most to least relevant."
                )
        elif originalOrder and not self.omit_priority_note:
            if self.terse_priority_note:
                terse_order = originalOrder.replace("Doc ", "").replace(" → ", ">")
                system_prompt += f"\nUse relevance order {terse_order}."
            elif self.compact_priority_note:
                system_prompt += f"\nEvidence priority: {originalOrder}."
            else:
                system_prompt += (
                    "\nNOTE: Please read the context in the following priority order: "
                    f"{originalOrder} to answer the question."
                )
        input_params = {"question": question, "reference": formatted_reference}
        input_params.update(**params)

        for k, v in input_params.items():
            system_prompt = system_prompt.replace("{" + k + "}", str(v))

        user_prompt = self.user_prompt
        for k, v in input_params.items():
            user_prompt = user_prompt.replace("{" + k + "}", str(v))

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if not self.is_openai:
                try:
                    input = self._get_tokenizer().apply_chat_template(
                        input, tokenize=False, add_generation_prompt=True
                    )
                except (AttributeError, TypeError, ValueError):
                    warnings.warn(
                        "The tokenizer does not provide a usable chat template; "
                        "falling back to plain prompt concatenation."
                    )
                    input = system_prompt + "\n\n" + user_prompt
        else:
            input = "\n\n".join(
                [prompt for prompt in [system_prompt, user_prompt] if prompt != ""]
            )

        result = input
        if not return_document_spans:
            return result

        document_spans = []
        if isinstance(result, str) and retrieval_result is not None:
            cursor = 0
            for doc_position, (doc_item, part) in enumerate(
                zip(retrieval_result, reference_parts)
            ):
                matched_part = part
                start = result.find(matched_part, cursor)
                if start < 0:
                    # Chat templates such as Llama-3 trim trailing whitespace
                    # from the complete system message. For the final document,
                    # locate the same content without its formatting newline.
                    matched_part = part.rstrip()
                    start = result.find(matched_part, cursor)
                if start < 0:
                    continue
                end = start + len(matched_part)
                document_spans.append(
                    {
                        "doc_id": doc_item.get("id"),
                        "doc_position": doc_position,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                cursor = end
        return result, document_spans

    def format_reference(
        self,
        retrieval_result,
        original_retrieval_result=None,
        return_parts=False,
    ):
        """Format reordered documents and retain stable document identities."""
        reference_parts = []
        doc_id_to_doc_number = {}
        relevance_rank = {}
        show_inline_rank = False
        if self.inline_relevance_rank and original_retrieval_result is not None:
            relevance_rank = {
                item.get("id"): rank
                for rank, item in enumerate(original_retrieval_result, start=1)
            }
            if not self.conditional_inline_rank:
                show_inline_rank = True
            else:
                reordered_ids = [item.get("id") for item in retrieval_result]
                original_ids = [item.get("id") for item in original_retrieval_result]
                show_inline_rank = reordered_ids != original_ids

        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title, separator, text = content.partition("\n")
            if not separator:
                text = ""

            doc_number = idx + 1
            doc_id = doc_item["id"]
            doc_id_to_doc_number[doc_id] = doc_number

            rank_suffix = ""
            if show_inline_rank and doc_id in relevance_rank:
                rank_suffix = f" [relevance={relevance_rank[doc_id]}]"

            if self.reference_template is not None:
                reference_parts.append(
                    self.reference_template.format(idx=idx, title=title, text=text)
                )
            elif self.stable_doc_format:
                reference_parts.append(f"{title}: {text}{rank_suffix}\n")
            elif self.compact_doc_format:
                reference_parts.append(f"[{doc_number}] {title}: {text}{rank_suffix}\n")
            else:
                reference_parts.append(
                    f"Doc {doc_number}(Title: {title}) {text}{rank_suffix}\n"
                )

        formatted_reference = "".join(reference_parts)
        if return_parts:
            return formatted_reference, doc_id_to_doc_number, reference_parts
        return formatted_reference, doc_id_to_doc_number

    def original_doc_order(self, original_results, doc_id_to_doc_number, sep=" → "):
        """Return the original relevance order after hotness reordering."""
        ordered_docs = []

        for doc_item in original_results:
            doc_id = doc_item.get("id")
            if doc_id is None:
                continue

            doc_number = doc_id_to_doc_number.get(doc_id)
            if doc_number is None:
                continue

            ordered_docs.append(f"Doc {doc_number}")

        if not ordered_docs:
            return ""

        original_order = sep.join(ordered_docs)

        max_doc_num = max(doc_id_to_doc_number.values(), default=0)
        default_order = sep.join(f"Doc {i}" for i in range(1, max_doc_num + 1))

        if original_order == default_order:
            return ""

        return original_order
