"""Dataset transformation helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np

from mnemorag.dataset.dataset import Dataset


def convert_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: convert_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_numpy(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def clean_prompt_images(prompt: Any) -> Any:
    """Remove image entries from a copied chat prompt before JSON export."""
    if not isinstance(prompt, list):
        return prompt
    cleaned = []
    for message in prompt:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [
                item
                for item in content
                if not isinstance(item, dict) or item.get("type") != "image"
            ]
        cleaned.append(copied)
    return cleaned


def filter_dataset(
    dataset: Dataset,
    predicate: Callable[[Any], bool] | None = None,
) -> Dataset:
    if predicate is None:
        return dataset
    return Dataset(
        config=dataset.config, data=[item for item in dataset if predicate(item)]
    )


def split_dataset(dataset: Dataset, labels: Sequence[Any]) -> dict[Any, Dataset]:
    if len(labels) != len(dataset):
        raise ValueError("one split label is required for each item")
    grouped: dict[Any, list[Any]] = {}
    for item, label in zip(dataset, labels):
        grouped.setdefault(label, []).append(item)
    return {
        label: Dataset(config=dataset.config, data=items)
        for label, items in grouped.items()
    }


def merge_dataset(
    datasets_by_label: dict[Any, Dataset], labels: Sequence[Any]
) -> Dataset:
    expected = sum(len(dataset) for dataset in datasets_by_label.values())
    if len(labels) != expected:
        raise ValueError("split labels do not match the supplied datasets")
    iterators = {label: iter(dataset) for label, dataset in datasets_by_label.items()}
    merged = [next(iterators[label]) for label in labels]
    first = next(iter(datasets_by_label.values()))
    return Dataset(config=first.config, data=merged)


def get_batch_dataset(dataset: Dataset, batch_size: int = 16) -> Iterator[Dataset]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for offset in range(0, len(dataset), batch_size):
        yield Dataset(
            config=dataset.config, data=dataset.data[offset : offset + batch_size]
        )


def merge_batch_dataset(dataset_list: Sequence[Dataset]) -> Dataset:
    if not dataset_list:
        raise ValueError("dataset_list must not be empty")
    items = [item for dataset in dataset_list for item in dataset]
    return Dataset(config=dataset_list[0].config, data=items)
