"""Configuration loading and resource initialization for MnemoRAG."""

from __future__ import annotations

import copy
import datetime as dt
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REQUIRED_PHYSICAL_CPU_CORES = 64
REQUIRED_FAISS_SEARCH_WORKERS = 64
REQUIRED_FAISS_OMP_THREADS = 1


def _cpu_topology(cpu_id: int) -> tuple[int, int]:
    topology_root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        socket_id = int((topology_root / "physical_package_id").read_text().strip())
        core_id = int((topology_root / "core_id").read_text().strip())
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Unable to resolve physical topology for CPU {cpu_id}"
        ) from error
    return socket_id, core_id


def _select_physical_cpu_ids(
    available_cpu_ids: list[int] | set[int],
) -> tuple[list[int], int]:
    """Select one hardware thread from 64 physical cores on one socket."""
    cores_by_socket: dict[int, dict[int, int]] = defaultdict(dict)
    for cpu_id in sorted(available_cpu_ids):
        socket_id, core_id = _cpu_topology(cpu_id)
        cores_by_socket[socket_id].setdefault(core_id, cpu_id)

    eligible = [
        socket_id
        for socket_id, cores in cores_by_socket.items()
        if len(cores) >= REQUIRED_PHYSICAL_CPU_CORES
    ]
    if not eligible:
        counts = {
            socket_id: len(cores)
            for socket_id, cores in sorted(cores_by_socket.items())
        }
        raise RuntimeError(
            "MnemoRAG requires 64 distinct physical CPU cores on one socket; "
            f"available physical cores by socket: {counts}"
        )

    socket_id = min(eligible)
    selected = sorted(cores_by_socket[socket_id].values())[:REQUIRED_PHYSICAL_CPU_CORES]
    return selected, socket_id


def _merge_dicts(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge configuration dictionaries without mutating inputs."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Resolved MnemoRAG configuration.

    Defaults are loaded from ``basic_config.yaml`` and then overridden by an
    optional YAML file and an optional dictionary, in that order.
    """

    def __init__(
        self,
        config_file_path: str | os.PathLike[str] | None = None,
        config_dict: Mapping[str, Any] | None = None,
    ) -> None:
        defaults = self._load_yaml(Path(__file__).with_name("basic_config.yaml"))
        file_config = (
            self._load_yaml(Path(config_file_path)) if config_file_path else {}
        )
        self.final_config = _merge_dicts(
            _merge_dicts(defaults, file_config), config_dict or {}
        )

        self._normalize()
        self._resolve_paths()
        self._init_cpu()
        self._init_device()
        self._set_seed()
        if not self.final_config.get("disable_save", False):
            self._prepare_dir()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Configuration root must be a mapping: {path}")
        return loaded

    def _normalize(self) -> None:
        split = self.final_config.get("split")
        if split is None:
            split = ["train", "dev", "test"]
        elif isinstance(split, str):
            split = [split]
        self.final_config["split"] = split

        framework = str(self.final_config.get("framework", "vllm")).lower()
        if framework != "vllm":
            raise ValueError("MnemoRAG supports framework='vllm' only")
        self.final_config["framework"] = framework

        if self.final_config.get("use_multi_retriever", False):
            raise ValueError("MnemoRAG supports one dense retriever per serving run")
        if self.final_config.get("use_sentence_transformer", False):
            raise ValueError("Use the native E5 encoder for reproducible retrieval")

    def _resolve_paths(self) -> None:
        config = self.final_config
        model_paths = config.get("model2path", {})
        pooling = config.get("model2pooling", {})
        indexes = config.get("method2index", {})

        dataset_name = config["dataset_name"]
        config.setdefault("dataset_path", str(Path(config["data_dir"]) / dataset_name))

        retrieval_method = config["retrieval_method"]
        if config.get("retrieval_model_path") is None:
            config["retrieval_model_path"] = model_paths.get(
                retrieval_method, retrieval_method
            )
        if config.get("retrieval_pooling_method") is None:
            config["retrieval_pooling_method"] = self._pooling_for(
                retrieval_method, pooling
            )
        if config.get("index_path") is None:
            config["index_path"] = indexes.get(retrieval_method)

        reranker_name = config.get("rerank_model_name")
        if reranker_name and config.get("rerank_model_path") is None:
            config["rerank_model_path"] = model_paths.get(reranker_name, reranker_name)
        if reranker_name and config.get("rerank_pooling_method") is None:
            config["rerank_pooling_method"] = self._pooling_for(reranker_name, pooling)

        generator_model = config["generator_model"]
        if config.get("generator_model_path") is None:
            config["generator_model_path"] = model_paths.get(
                generator_model, generator_model
            )

    @staticmethod
    def _pooling_for(model_name: str, model2pooling: Mapping[str, str]) -> str:
        lowered = model_name.lower()
        return next(
            (value for key, value in model2pooling.items() if key.lower() in lowered),
            "mean",
        )

    def _init_cpu(self) -> None:
        config = self.final_config
        cpu_threads = int(config.get("cpu_threads", REQUIRED_PHYSICAL_CPU_CORES))
        if cpu_threads != REQUIRED_PHYSICAL_CPU_CORES:
            raise ValueError("MnemoRAG requires cpu_threads=64 physical cores")

        selected, socket_id = _select_physical_cpu_ids(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(selected))
        config.update(
            cpu_threads=REQUIRED_PHYSICAL_CPU_CORES,
            cpu_affinity=selected,
            physical_core_count=REQUIRED_PHYSICAL_CPU_CORES,
            cpu_socket_ids=[socket_id],
            smt_siblings_selected=False,
        )

        omp_threads = int(config.get("faiss_omp_threads") or REQUIRED_FAISS_OMP_THREADS)
        workers = int(
            config.get("faiss_search_workers") or REQUIRED_FAISS_SEARCH_WORKERS
        )
        max_codes = int(config.get("faiss_max_codes") or 0)
        if omp_threads != REQUIRED_FAISS_OMP_THREADS:
            raise ValueError("MnemoRAG requires faiss_omp_threads=1")
        if workers != REQUIRED_FAISS_SEARCH_WORKERS:
            raise ValueError("MnemoRAG requires faiss_search_workers=64")
        if max_codes < 0:
            raise ValueError("faiss_max_codes must be nonnegative")

        config.update(
            faiss_omp_threads=omp_threads,
            faiss_search_workers=workers,
            faiss_max_codes=max_codes,
        )
        environment = {
            "MNEMORAG_FAISS_OMP_THREADS": omp_threads,
            "MNEMORAG_FAISS_SEARCH_WORKERS": workers,
            "MNEMORAG_FAISS_MAX_CODES": max_codes,
            "MNEMORAG_FAISS_BATCH_FULL": 0,
            "OMP_NUM_THREADS": omp_threads,
            "OMP_THREAD_LIMIT": omp_threads,
            "MKL_NUM_THREADS": omp_threads,
            "OPENBLAS_NUM_THREADS": omp_threads,
            "NUMEXPR_NUM_THREADS": omp_threads,
        }
        os.environ.update({name: str(value) for name, value in environment.items()})

    def _init_device(self) -> None:
        gpu_id = self.final_config.get("gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_count = torch.cuda.device_count()
        self.final_config["gpu_num"] = gpu_count
        self.final_config["device"] = "cuda" if gpu_count else "cpu"

    def _set_seed(self) -> None:
        try:
            seed = int(self.final_config.get("seed", 2024))
        except (TypeError, ValueError):
            seed = 2024
        self.final_config["seed"] = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _prepare_dir(self) -> None:
        timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
        directory = Path(self.final_config["save_dir"]) / (
            f"{self.final_config['dataset_name']}_{timestamp}_"
            f"{self.final_config['save_note']}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        self.final_config["save_dir"] = str(directory)
        with (directory / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.final_config, stream, sort_keys=False)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("configuration keys must be strings")
        self.final_config[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.final_config.get(key)

    def __contains__(self, key: str) -> bool:
        if not isinstance(key, str):
            raise TypeError("configuration keys must be strings")
        return key in self.final_config

    def __getattr__(self, key: str) -> Any:
        if "final_config" in self.__dict__ and key in self.final_config:
            return self.final_config[key]
        raise AttributeError(f"Config has no attribute {key!r}")

    def __repr__(self) -> str:
        return repr(self.final_config)
