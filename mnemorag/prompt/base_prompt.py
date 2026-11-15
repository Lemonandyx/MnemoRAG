from transformers import AutoTokenizer


class PromptTemplate:
    placeholders = ["reference", "question"]
    base_system_prompt = (
        "Answer the question based on the given document. "
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}"
    )
    base_user_prompt = "Question: {question}\nAnswer:"

    def __init__(
        self,
        config,
        system_prompt="",
        user_prompt="",
        reference_template=None,
        enable_chat=True,
    ):

        self.config = config
        self.max_input_len = config["generator_max_input_len"]
        self.generator_path = config["generator_model_path"]
        self.is_chat = True

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
        self.tokenizer = None

    def bind_tokenizer(self, tokenizer):
        """Reuse a generator-owned tokenizer when the pipeline provides one."""
        self.tokenizer = tokenizer

    def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.generator_path, trust_remote_code=True
            )
        return self.tokenizer

    def get_string(
        self,
        question=None,
        retrieval_result=None,
        original_retrieval_result=None,
        formatted_reference=None,
        previous_gen=None,
        return_document_spans=False,
        **params,
    ):
        reference_parts = []
        doc_id_to_doc_number = {}
        if formatted_reference is None:
            if retrieval_result is not None:
                (
                    formatted_reference,
                    doc_id_to_doc_number,
                    reference_parts,
                ) = self.format_reference(
                    retrieval_result,
                    return_parts=True,
                )
            else:
                formatted_reference = ""

        system_prompt = self.system_prompt
        if previous_gen is not None and previous_gen.strip() != "":
            system_prompt += (
                "\n\nKnown information from previous iteration:\n"
                "{previous_gen}\n"
                "Note: Do not repeat the draft answer verbatim unless no improvement is possible.\n"
            )
        if original_retrieval_result is not None:
            original_order = self.original_doc_order(
                original_retrieval_result, doc_id_to_doc_number
            )
        else:
            original_order = ""
        if original_order:
            system_prompt += (
                "\nNOTE: Please read the context in the following priority order: "
                f"{original_order} to answer the question."
            )
        input_params = {
            "question": question,
            "reference": formatted_reference,
            "previous_gen": previous_gen,
        }
        input_params.update(**params)

        system_prompt = system_prompt.format_map(input_params)
        user_prompt = self.user_prompt.format_map(input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            input = self._get_tokenizer().apply_chat_template(
                input, tokenize=False, add_generation_prompt=True
            )
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
        return_parts=False,
    ):
        """Format reordered documents and retain stable document identities."""
        reference_parts = []
        doc_id_to_doc_number = {}
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title, separator, text = content.partition("\n")
            if not separator:
                text = ""

            doc_number = idx + 1
            doc_id = doc_item["id"]
            doc_id_to_doc_number[doc_id] = doc_number

            if self.reference_template is not None:
                reference_parts.append(
                    self.reference_template.format(idx=idx, title=title, text=text)
                )
            else:
                document = f"{title} {text}".strip()
                reference_parts.append(f"Doc{doc_number}: {document}\n")

        formatted_reference = "".join(reference_parts)
        if return_parts:
            return formatted_reference, doc_id_to_doc_number, reference_parts
        return formatted_reference, doc_id_to_doc_number

    def original_doc_order(self, original_results, doc_id_to_doc_number, sep=">"):
        """Return the original relevance order after hotness reordering."""
        ordered_docs = []

        for doc_item in original_results:
            doc_id = doc_item.get("id")
            if doc_id is None:
                continue

            doc_number = doc_id_to_doc_number.get(doc_id)
            if doc_number is None:
                continue

            ordered_docs.append(f"Doc{doc_number}")

        if not ordered_docs:
            return ""

        original_order = sep.join(ordered_docs)

        max_doc_num = max(doc_id_to_doc_number.values(), default=0)
        default_order = sep.join(f"Doc{i}" for i in range(1, max_doc_num + 1))

        if original_order == default_order:
            return ""

        return original_order
