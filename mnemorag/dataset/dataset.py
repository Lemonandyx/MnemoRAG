"""Dataset containers used by MnemoRAG pipelines and evaluators."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import datasets


class Item:
    """One QA request together with fields produced during serving."""

    _CORE_FIELDS = {
        "id",
        "question",
        "golden_answers",
        "choices",
        "metadata",
        "output",
        "data",
    }

    def __init__(self, item_dict: Mapping[str, Any]) -> None:
        source = dict(item_dict)
        object.__setattr__(self, "id", source.get("id"))
        object.__setattr__(self, "question", source.get("question"))
        object.__setattr__(self, "golden_answers", source.get("golden_answers", []))
        object.__setattr__(self, "choices", source.get("choices", []))
        object.__setattr__(self, "metadata", source.get("metadata", {}))
        object.__setattr__(self, "output", dict(source.get("output", {})))
        object.__setattr__(self, "data", source)

    def update_output(self, key: str, value: Any) -> None:
        if key in self._CORE_FIELDS:
            raise AttributeError(f"core field {key!r} cannot be replaced")
        self.output[key] = value

    def update_evaluation_score(self, metric_name: str, metric_score: float) -> None:
        self.output.setdefault("metric_score", {})[metric_name] = metric_score

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __getattr__(self, key: str) -> Any:
        output = object.__getattribute__(self, "output")
        if key in output:
            return output[key]
        data = object.__getattribute__(self, "data")
        if key in data:
            return data[key]
        raise AttributeError(f"Item has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._CORE_FIELDS:
            object.__setattr__(self, key, value)
        else:
            self.update_output(key, value)

    def to_dict(self) -> dict[str, Any]:
        from mnemorag.dataset.utils import clean_prompt_images, convert_numpy

        result = {
            key: convert_numpy(value)
            for key, value in self.data.items()
            if key != "output"
        }
        output = convert_numpy(self.output)
        if "prompt" in output:
            output["prompt"] = clean_prompt_images(output["prompt"])
        result["output"] = output
        return result

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class Dataset:
    """Ordered collection of :class:`Item` objects."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        dataset_path: str | Path | None = None,
        data: Sequence[Mapping[str, Any] | Item] | None = None,
        sample_num: int | None = None,
        random_sample: bool = False,
    ) -> None:
        self.config = config
        self.dataset_name = (
            config["dataset_name"] if config and "dataset_name" in config else "default"
        )
        self.dataset_path = str(dataset_path) if dataset_path is not None else None
        self.sample_num = sample_num
        self.random_sample = random_sample

        if data is None:
            if self.dataset_path is None:
                raise ValueError("dataset_path is required when data is not supplied")
            items = self._load_data(Path(self.dataset_path))
        else:
            items = [item if isinstance(item, Item) else Item(item) for item in data]
        self.data = self._sample(items)

    def _load_data(self, path: Path) -> list[Item]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as stream:
                return [Item(json.loads(line)) for line in stream if line.strip()]
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, list):
                raise TypeError("JSON datasets must contain a top-level list")
            return [Item(entry) for entry in payload]
        if suffix == ".parquet":
            table = datasets.load_dataset(
                "parquet", data_files=str(path), split="train"
            )
            return [Item(entry) for entry in table]
        raise ValueError(f"Unsupported dataset format: {path}")

    def _sample(self, items: list[Item]) -> list[Item]:
        if self.sample_num is None:
            return items
        count = int(self.sample_num)
        if count < 0:
            raise ValueError("sample_num must be nonnegative")
        if self.random_sample:
            if count > len(items):
                raise ValueError("sample_num exceeds the dataset size")
            return random.sample(items, count)
        return items[:count]

    def update_output(self, key: str, values: Sequence[Any]) -> None:
        if len(values) != len(self.data):
            raise ValueError("one output value is required for each dataset item")
        for item, value in zip(self.data, values):
            item.update_output(key, value)

    @property
    def question(self) -> list[str | None]:
        return [item.question for item in self.data]

    @property
    def golden_answers(self) -> list[list[str]]:
        return [item.golden_answers for item in self.data]

    @property
    def id(self) -> list[str | None]:
        return [item.id for item in self.data]

    @property
    def output(self) -> list[dict[str, Any]]:
        return [item.output for item in self.data]

    def get_batch_data(self, key: str, batch_size: int) -> Iterator[list[Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for offset in range(0, len(self.data), batch_size):
            yield [item[key] for item in self.data[offset : offset + batch_size]]

    def get_attr_data(self, key: str) -> list[Any]:
        return [item[key] for item in self.data]

    def __getattr__(self, key: str) -> list[Any]:
        if key.startswith("__"):
            raise AttributeError(key)
        return [getattr(item, key) for item in self.data]

    def __getitem__(self, index: int | slice) -> Item | list[Item]:
        return self.data[index]

    def __iter__(self) -> Iterator[Item]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def save(self, save_path: str | Path) -> None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(
                [item.to_dict() for item in self.data],
                stream,
                indent=2,
                ensure_ascii=False,
            )

    def __str__(self) -> str:
        return f"Dataset {self.dataset_name!r} with {len(self)} items"
