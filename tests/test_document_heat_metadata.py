import asyncio
from types import SimpleNamespace

import pytest

from mnemorag.dataset.dataset import Dataset
from mnemorag.generator.generator import VLLMGenerator
from mnemorag.pipeline.pipeline import AsyncPipeline
from mnemorag.pipeline.utils import DocumentPool
from mnemorag.prompt.base_prompt import PromptTemplate


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        body = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return body + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            input_ids.insert(0, 0)
            offsets.insert(0, (0, 0))
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class TrimmingChatTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        body = "".join(
            f"<{message['role']}>{message['content'].strip()}" for message in messages
        )
        return body + ("<assistant>" if add_generation_prompt else "")


def make_prompt_template():
    template = PromptTemplate.__new__(PromptTemplate)
    template.system_prompt = "Documents:\n{reference}"
    template.user_prompt = "Question: {question}"
    template.reference_template = None
    template.is_chat = True
    template.enable_chat = True
    template.is_openai = False
    template.tokenizer = CharacterTokenizer()
    template.inline_relevance_rank = False
    template.conditional_inline_rank = False
    template.strong_priority_note = False
    template.concise_strong_priority_note = False
    template.terse_priority_note = False
    template.compact_priority_note = False
    template.omit_priority_note = True
    template.stable_doc_format = False
    template.compact_doc_format = False
    return template


def test_prompt_reports_exact_document_character_spans_after_chat_template():
    template = make_prompt_template()
    docs = [
        {"id": 7, "contents": "Alpha\nfirst fact"},
        {"id": 9, "contents": "Beta\nsecond fact"},
    ]

    prompt, spans = template.get_string(
        question="Which?",
        retrieval_result=docs,
        return_document_spans=True,
    )

    assert [span["doc_id"] for span in spans] == [7, 9]
    assert [span["doc_position"] for span in spans] == [0, 1]
    assert prompt[spans[0]["char_start"] : spans[0]["char_end"]] == (
        "Doc 1(Title: Alpha) first fact\n"
    )
    assert prompt[spans[1]["char_start"] : spans[1]["char_end"]] == (
        "Doc 2(Title: Beta) second fact\n"
    )


def test_prompt_reports_final_document_span_when_chat_template_trims_newline():
    template = make_prompt_template()
    template.tokenizer = TrimmingChatTokenizer()
    docs = [{"id": 7, "contents": "Alpha\nfirst fact"}]

    prompt, spans = template.get_string(
        question="Which?",
        retrieval_result=docs,
        return_document_spans=True,
    )

    assert len(spans) == 1
    span = spans[0]
    assert prompt[span["char_start"] : span["char_end"]] == (
        "Doc 1(Title: Alpha) first fact"
    )


def test_pipeline_uses_retrieval_index_for_document_heat():
    pipeline = AsyncPipeline.__new__(AsyncPipeline)
    pipeline.enable_document_kv_heat = True
    pipeline.prompt_template = make_prompt_template()
    pipeline.document_pool = DocumentPool(num_documents=16)
    pipeline.document_pool.update([3, 3, 3])
    docs = [{"id": "external-document-id", "contents": "Alpha\nfirst fact"}]

    _, spans = pipeline._build_prompt_with_document_heat(
        question="Which?",
        docs=docs,
        retrieval_idxs=[3],
    )

    assert spans[0]["heat"] == 3
    assert pipeline.get_document_heat_metadata_stats() == {
        "requests": 1,
        "spans": 1,
        "above_one_requests": 1,
        "above_one_spans": 1,
        "max_heat": 3.0,
    }


def make_heat_pipeline():
    pipeline = AsyncPipeline.__new__(AsyncPipeline)
    pipeline.enable_document_kv_heat = True
    pipeline.prompt_template = make_prompt_template()
    pipeline.document_pool = DocumentPool(num_documents=16)
    pipeline.config = {
        "retrieval_batch_size": 2,
        "irg_max_iter": 2,
        "irg_min_iter": 1,
        "irg_early_stop": False,
        "generation_params": {"max_tokens": 8},
        "irg_round_max_tokens": 8,
        "recomp_compression_max_tokens": 8,
        "recomp_answer_max_tokens": 8,
    }
    pipeline.evaluate = lambda dataset, **_: dataset
    return pipeline


def test_irg_passes_document_heat_metadata_on_every_round():
    pipeline = make_heat_pipeline()
    generation_calls = []

    async def retrieve(queries):
        return (
            [[{"id": 3, "contents": "Alpha\nshared fact"}] for _ in queries],
            [[1.0] for _ in queries],
            [[3] for _ in queries],
        )

    async def generate(
        prompts,
        document_heat_spans=None,
        return_completion_times=False,
        **_,
    ):
        generation_calls.append((list(prompts), document_heat_spans))
        outputs = [f"draft-{len(generation_calls)}" for _ in prompts]
        if return_completion_times:
            return outputs, [float(len(generation_calls)) for _ in prompts]
        return outputs

    pipeline._retrieve_async = retrieve
    pipeline._generate_async_safe = generate
    dataset = Dataset(data=[{"question": "q0"}, {"question": "q1"}])

    result = asyncio.run(pipeline.run_stream_irg(dataset, do_eval=False))

    assert len(generation_calls) == 2
    assert all(
        len(request_spans) == 1
        for _, batch_spans in generation_calls
        for request_spans in batch_spans
    )
    assert generation_calls[0][1][0][0]["heat"] == 2
    assert generation_calls[1][1][0][0]["heat"] == 4
    assert pipeline.get_document_heat_metadata_stats() == {
        "requests": 4,
        "spans": 4,
        "above_one_requests": 4,
        "above_one_spans": 4,
        "max_heat": 4.0,
    }
    assert all(item.output["pred"] == "draft-2" for item in result)


def test_recomp_passes_heat_for_compression_and_compressed_evidence():
    pipeline = make_heat_pipeline()
    generation_calls = []

    async def retrieve(queries):
        return (
            [[{"id": 5, "contents": "Beta\nshared fact"}] for _ in queries],
            [[1.0] for _ in queries],
            [[5] for _ in queries],
        )

    async def generate(
        prompts,
        document_heat_spans=None,
        return_completion_times=False,
        **_,
    ):
        generation_calls.append((list(prompts), document_heat_spans))
        outputs = [
            "compressed shared evidence" if len(generation_calls) == 1 else "answer"
            for _ in prompts
        ]
        if return_completion_times:
            return outputs, [float(len(generation_calls)) for _ in prompts]
        return outputs

    pipeline._retrieve_async = retrieve
    pipeline._generate_async_safe = generate
    dataset = Dataset(data=[{"question": "q0"}, {"question": "q1"}])

    result = asyncio.run(pipeline.run_stream_recomp(dataset, do_eval=False))

    assert len(generation_calls) == 2
    compression_prompts, compression_spans = generation_calls[0]
    answer_prompts, answer_spans = generation_calls[1]
    assert all(len(spans) == 1 for spans in compression_spans)
    assert all(len(spans) == 1 for spans in answer_spans)
    assert (
        compression_prompts[0][
            compression_spans[0][0]["char_start"] : compression_spans[0][0]["char_end"]
        ]
        == "Document 1: Beta\nshared fact"
    )
    assert (
        answer_prompts[0][
            answer_spans[0][0]["char_start"] : answer_spans[0][0]["char_end"]
        ]
        == "compressed shared evidence"
    )
    assert pipeline.get_document_heat_metadata_stats() == {
        "requests": 4,
        "spans": 4,
        "above_one_requests": 4,
        "above_one_spans": 4,
        "max_heat": 2.0,
    }
    assert all(item.output["pred"] == "answer" for item in result)


def test_generator_attaches_heat_to_the_same_pretokenized_prompt():
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.tokenizer = CharacterTokenizer()
    sampling_params = SimpleNamespace(extra_args={"preserved": 1})
    prompt = "prefix-document-suffix"

    prepared_prompt, prepared_params = generator._prepare_document_heat_request(
        prompt,
        [{"char_start": 7, "char_end": 15, "heat": 11}],
        sampling_params,
    )

    assert prepared_prompt == {"prompt_token_ids": [0, *[ord(char) for char in prompt]]}
    assert prepared_params.extra_args == {
        "preserved": 1,
        "mnemorag_document_heat_spans": [[8, 16, 11.0]],
    }
    assert sampling_params.extra_args == {"preserved": 1}


def test_generator_rejects_misaligned_batch_metadata():
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.use_persistent_async_engine = True

    with pytest.raises(ValueError, match="one entry per prompt"):
        import asyncio

        asyncio.run(
            generator.generate_async(["first", "second"], document_heat_spans=[None])
        )
