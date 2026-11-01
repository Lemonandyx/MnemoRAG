from mnemorag import Config, Dataset, MnemoRAGPipeline
from mnemorag.pipeline.pipeline import AsyncPipeline


def test_public_api_exposes_mnemorag_pipeline():
    assert issubclass(MnemoRAGPipeline, AsyncPipeline)
    assert Config.__name__ == "Config"
    assert Dataset.__name__ == "Dataset"
