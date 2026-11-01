from types import SimpleNamespace

from mnemorag.pipeline.pipeline import BasicPipeline
from mnemorag.prompt.base_prompt import PromptTemplate


def _config():
    return {
        "framework": "vllm",
        "generator_max_input_len": 2048,
        "generator_model_path": "/models/test",
        "is_reasoning": False,
    }


def test_prompt_tokenizer_is_lazy(monkeypatch):
    monkeypatch.setattr(
        "mnemorag.prompt.base_prompt.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_name_or_path="test-instruct"),
    )
    calls = []
    tokenizer = object()

    def load_tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setattr(
        "mnemorag.prompt.base_prompt.AutoTokenizer.from_pretrained",
        load_tokenizer,
    )
    template = PromptTemplate(_config())

    assert template.tokenizer is None
    assert calls == []
    assert template._get_tokenizer() is tokenizer
    assert len(calls) == 1


def test_pipeline_reuses_generator_tokenizer():
    shared_tokenizer = object()
    template = SimpleNamespace(tokenizer=None)
    template.bind_tokenizer = lambda tokenizer: setattr(
        template, "tokenizer", tokenizer
    )
    pipeline = BasicPipeline.__new__(BasicPipeline)
    pipeline.prompt_template = template
    pipeline.generator = SimpleNamespace(tokenizer=shared_tokenizer)

    pipeline._bind_prompt_tokenizer()

    assert template.tokenizer is shared_tokenizer
