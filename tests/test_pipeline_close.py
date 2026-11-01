from mnemorag.pipeline.pipeline import BasicPipeline


class _Retriever:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    def close(self):
        self.events.append("retriever")
        if self.fail:
            raise RuntimeError("retriever close failed")


class _Generator:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append("generator")


def _pipeline(retriever):
    events = retriever.events
    pipeline = object.__new__(BasicPipeline)
    pipeline.retriever = retriever
    pipeline.generator = _Generator(events)
    return pipeline


def test_pipeline_close_releases_both_components():
    events = []
    _pipeline(_Retriever(events)).close()

    assert events == ["retriever", "generator"]


def test_pipeline_close_still_releases_generator_on_retriever_error():
    events = []
    pipeline = _pipeline(_Retriever(events, fail=True))

    try:
        pipeline.close()
    except RuntimeError as error:
        assert str(error) == "retriever close failed"
    else:
        raise AssertionError("expected retriever close failure")

    assert events == ["retriever", "generator"]
