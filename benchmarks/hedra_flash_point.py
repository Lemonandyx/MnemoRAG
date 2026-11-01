#!/usr/bin/env python3
"""Run one isolated HedraRAG or Hedra-provided FlashRAG comparison point.

The two systems retain their own executor and retrieval implementations.  The
request state machines below only normalize prompts, IRG query updates, RECOMP
stages, and final-output selection to the MnemoRAG evaluation contract.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import platform
import random
import re
import statistics
import string
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .benchmark_point import (
        canonical_sha256,
        current_numa_policy,
        resource_preflight,
        validate_physical_cpu_ids,
    )
    from .experiment_validation import (
        CANONICAL_SYSTEM_PROMPT,
        CANONICAL_USER_PROMPT,
        expected_prompt_contract,
    )
    from .fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        build_trace_manifest,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
        summarize_latency_slo,
        validate_trace_workload,
    )
    from .runtime_resource_monitor import RuntimeResourceMonitor
    from .vllm_contract import validate_effective_vllm_contract
except ImportError:  # Direct script execution.
    from benchmark_point import (
        canonical_sha256,
        current_numa_policy,
        resource_preflight,
        validate_physical_cpu_ids,
    )
    from experiment_validation import (
        CANONICAL_SYSTEM_PROMPT,
        CANONICAL_USER_PROMPT,
        expected_prompt_contract,
    )
    from fair_workload import (
        ArrivalTraceSpec,
        build_arrival_offsets,
        build_trace_manifest,
        file_sha256,
        load_trace_bundle,
        load_trace_sample_indices,
        summarize_latency_slo,
        validate_trace_workload,
    )
    from runtime_resource_monitor import RuntimeResourceMonitor
    from vllm_contract import validate_effective_vllm_contract

if os.environ.get("HEDRA_FAULTHANDLER") == "1":
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
HEDRA_ROOT = Path(
    os.environ.get("HEDRARAG_ROOT", PROJECT_ROOT / "third_party/HedraRAG")
).resolve()
HEDRA_AE_UPSTREAM = "https://github.com/Leo9660/HedraRAG_AE"
HEDRA_AE_COMMIT = "50a760364c954bf153c03fdcb94bf4555bfe44f0"
HEDRA_FAISS_ROOT = Path(
    os.environ.get("HEDRARAG_FAISS_ROOT", HEDRA_ROOT / "third_party/faiss")
).resolve()
HEDRA_FAISS_PYTHON = HEDRA_FAISS_ROOT / "python"
HEDRA_FAISS_LIBRARY = HEDRA_FAISS_ROOT
BASELINE_PROMPT_ORDERING_MODES = {
    "flashrag": "original",
    "hedrarag": "retrieval",
}
DATASET_PATHS = {
    "nq": ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    "hotpotqa": ASSET_ROOT / "qa/datasets/hotpotqa/train.jsonl",
    "webquestions": ASSET_ROOT / "qa/datasets/web_questions/train.jsonl",
    "squad": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
    "2wikimultihopqa": ASSET_ROOT / "qa/datasets/2wikimultihopqa/train.jsonl",
}


def install_hedra_engine_args_contract(generator_module, args) -> list[dict]:
    """Inject common scheduler limits into Hedra's otherwise fixed EngineArgs."""
    original_engine_args = generator_module.EngineArgs
    calls: list[dict] = []
    required = {
        "max_num_seqs": int(args.serving_batch_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_prefix_caching": True,
    }
    if int(args.max_num_batched_tokens) > 0:
        required["max_num_batched_tokens"] = int(args.max_num_batched_tokens)

    def contracted_engine_args(*positional, **keywords):
        if positional:
            raise RuntimeError("Hedra EngineArgs must use keyword arguments")
        conflicts = {
            name: {"required": value, "observed": keywords[name]}
            for name, value in required.items()
            if name in keywords and keywords[name] != value
        }
        if conflicts:
            raise RuntimeError(f"Hedra EngineArgs conflict: {conflicts}")
        keywords.update(required)
        calls.append(dict(keywords))
        return original_engine_args(**keywords)

    generator_module.EngineArgs = contracted_engine_args
    return calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("hedrarag", "flashrag"), required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="nq")
    parser.add_argument(
        "--workflow", choices=("onepass", "recomp", "irg"), default="onepass"
    )
    parser.add_argument("--num-requests", type=int, default=1024)
    parser.add_argument("--arrival-rate", type=float, default=24.0)
    parser.add_argument("--arrival-burstiness", type=float, default=1.0)
    parser.add_argument(
        "--normalize-arrival-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--arrival-trace",
        type=Path,
        default=None,
        help="Canonical .npz arrival trace shared by every compared system.",
    )
    parser.add_argument(
        "--online-warmup-indices",
        type=Path,
        help=(
            "Disjoint request indices executed on the same engine before "
            "measurement; warmup latency and quality are excluded."
        ),
    )
    parser.add_argument("--latency-slo-seconds", type=float, default=10.0)
    parser.add_argument("--campaign-group-id", default="standalone")
    parser.add_argument("--campaign-repeat", type=int, default=0)
    parser.add_argument("--campaign-order-position", type=int, default=0)
    parser.add_argument("--topk", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--nprobe", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--serving-batch-size", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument(
        "--hot-cluster-csv",
        type=Path,
        default=None,
        help=(
            "Audited hot-cluster ordering derived from disjoint online warmup "
            "requests. Required for HedraRAG compatibility measurements."
        ),
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--require-clean-resources", action="store_true")
    parser.add_argument("--max-preflight-cpu-busy-percent", type=float, default=20.0)
    parser.add_argument(
        "--max-preflight-cpu-max-busy-percent", type=float, default=50.0
    )
    parser.add_argument("--max-preflight-gpu-memory-mib", type=float, default=256.0)
    parser.add_argument("--max-preflight-gpu-util-percent", type=float, default=10.0)
    parser.add_argument("--max-preflight-load-per-cpu", type=float, default=0.10)
    parser.add_argument("--runtime-resource-sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-runtime-external-cpu-busy-percent",
        type=float,
        default=5.0,
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--recomp-compression-max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM allocation fraction. Defaults to 0.90 for CPU-retrieval "
            "FlashRAG and 0.75 for HedraRAG, which reserves a 4 GiB GPU "
            "retrieval cache."
        ),
    )
    parser.add_argument("--irg-rounds", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluation/baselines",
    )
    return parser.parse_args()


def configure(args: argparse.Namespace) -> list[int]:
    configured_gpu_fraction = getattr(args, "gpu_memory_utilization", None)
    system = getattr(args, "system", "flashrag")
    if configured_gpu_fraction is None:
        args.gpu_memory_utilization = 0.75 if system == "hedrarag" else 0.90
        args.gpu_memory_policy = "system_native_default"
    else:
        args.gpu_memory_policy = "explicit_override"
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads != 64 or len(available) != args.cpu_threads:
        raise ValueError(
            "baseline points require an externally pinned set of exactly "
            "64 logical CPUs"
        )
    selected = available
    if selected != list(range(64)):
        raise RuntimeError("baseline CPU affinity must be exactly physical CPUs 0-63")
    os.sched_setaffinity(0, set(selected))
    args.cpu_topology = validate_physical_cpu_ids(selected)
    if args.cpu_topology["physical_core_count"] != 64:
        raise RuntimeError("CPU affinity must contain 64 physical cores")
    if args.cpu_topology["smt_siblings_selected"]:
        raise RuntimeError("CPU affinity must not include SMT siblings")
    if args.cpu_topology["socket_ids"] != [0]:
        raise RuntimeError("all 64 physical cores must belong to socket 0")
    if args.cpu_topology["numa_node_ids"] != [0, 1, 2, 3]:
        raise RuntimeError("baseline CPU affinity must span NUMA nodes 0-3")
    if args.cpu_topology["numa_node_core_counts"] != {
        "0": 16,
        "1": 16,
        "2": 16,
        "3": 16,
    }:
        raise RuntimeError("each of NUMA nodes 0-3 must contribute 16 cores")
    if args.cpu_topology["socket_core_counts"] != {"0": 64}:
        raise RuntimeError("socket 0 must contribute all 64 physical cores")
    args.numa_policy = current_numa_policy()
    if args.numa_policy.get("policy") != "interleave":
        raise RuntimeError("NUMA memory policy must interleave NUMA nodes 0-3")
    expected_numa_text = "0 1 2 3"
    if args.numa_policy.get("physcpubind") != " ".join(map(str, range(64))):
        raise RuntimeError("numactl physcpubind must be CPUs 0-63")
    for field in ("cpubind", "nodebind", "interleavemask"):
        if args.numa_policy.get(field) != expected_numa_text:
            raise RuntimeError(f"numactl {field} must be NUMA nodes 0-3")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # The HedraRAG config initializes CUDA before vLLM constructs its V1 core.
    # Fork would inherit an initialized CUDA runtime and fail; spawn preserves
    # the native multiprocess engine without changing executor semantics.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    args.resource_preflight = resource_preflight(args, selected)
    print(
        "[MnemoRAG] Baseline resource preflight: "
        + json.dumps(args.resource_preflight, sort_keys=True),
        flush=True,
    )
    return selected


def ensure_native_hedra_faiss(args: argparse.Namespace) -> None:
    """Restart Hedra points with the artifact's custom FAISS on the loader path.

    FlashRAG points deliberately stay on the shared FAISS 1.8 environment so
    the custom Hedra extensions cannot alter the baseline implementation.
    """
    if args.system != "hedrarag" or os.environ.get("HEDRA_CUSTOM_FAISS_ACTIVE") == "1":
        return
    if not HEDRA_FAISS_PYTHON.exists() or not HEDRA_FAISS_LIBRARY.exists():
        raise FileNotFoundError("HedraRAG custom FAISS build is missing")
    env = os.environ.copy()
    env["HEDRA_CUSTOM_FAISS_ACTIVE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HEDRA_FAISS_PYTHON), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(HEDRA_FAISS_LIBRARY), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def load_sample(
    path: Path,
    count: int,
    seed: int,
    *,
    explicit_indices=None,
) -> tuple[list[dict], list[int], int]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if explicit_indices is None:
        if count > len(rows):
            raise ValueError(f"Requested {count} rows from {len(rows)} available")
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(rows)), count))
    else:
        indices = [int(index) for index in explicit_indices]
        if len(indices) != count:
            raise ValueError(
                f"Explicit sample has {len(indices)} rows; expected {count}"
            )
        if any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError("Explicit sample index is outside the dataset")
    return [rows[index] for index in indices], indices, len(rows)


def normalize(text: str) -> list[str]:
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).split()


def answers(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def runtime_versions() -> dict[str, object]:
    versions: dict[str, object] = {"python": platform.python_version()}
    for package in ("vllm", "torch", "faiss", "transformers"):
        try:
            module = __import__(package)
            versions[package] = {
                "version": str(getattr(module, "__version__", "unknown")),
                "path": str(Path(module.__file__).resolve()),
            }
        except Exception as error:  # pragma: no cover - diagnostic only
            versions[package] = {"error": str(error)}
    for name, root in (
        ("mnemorag_git_commit", PROJECT_ROOT),
        ("hedrarag_git_commit", HEDRA_ROOT),
    ):
        try:
            versions[name] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            versions[name] = "unavailable"
    return versions


def implementation_fidelity(
    system: str,
    hedra_root: Path = HEDRA_ROOT,
    *,
    topk: int | None = None,
    nprobe: int | None = None,
) -> dict[str, object]:
    if system == "flashrag":
        executor_path = hedra_root / "heterag/executor/sequential_executor.py"
        return {
            "scheduler": "hedrarag_artifact_sequential_executor",
            "retriever": "flashrag_cpu_faiss",
            "upstream_repository": HEDRA_AE_UPSTREAM,
            "upstream_commit": HEDRA_AE_COMMIT,
            "upstream_driver": "evaluation/fig_12/FlashRAG_test.sh",
            "executor_class": "heterag.executor.sequential_executor.SequentialExecutor",
            "local_executor_path": str(executor_path),
            "local_executor_sha256": file_sha256(executor_path),
            "experiment_parameter_overrides": {
                "topk": topk,
                "nprobe": nprobe,
                "reason": "shared formal experiment contract",
            },
            "native_cuda_sources_present": None,
            "headline_eligible": True,
        }

    cuda_sources = (
        hedra_root / "heterag/retriever/gpu_retrieval_backend/search_kernel.cu",
        hedra_root / "heterag/retriever/gpu_retrieval_backend/search_bind.cpp",
    )
    native_present = all(path.is_file() for path in cuda_sources)
    return {
        "scheduler": "hedrarag_artifact_scheduler",
        "upstream_repository": HEDRA_AE_UPSTREAM,
        "upstream_commit": HEDRA_AE_COMMIT,
        "experiment_parameter_overrides": {
            "topk": topk,
            "nprobe": nprobe,
            "reason": "shared formal experiment contract",
        },
        "retriever": (
            "hedrarag_native_cuda"
            if native_present
            else "a100_torch_faiss_compatibility_backend"
        ),
        "native_cuda_sources_present": native_present,
        "expected_native_cuda_sources": [str(path) for path in cuda_sources],
        "compatibility_backend": str(
            hedra_root / "heterag/retriever/a100_torch_backend.py"
        ),
        "headline_eligible": native_present,
    }


def score_prediction(prediction: str, golden) -> dict[str, float]:
    pred = normalize(prediction)
    pred_text = " ".join(pred)
    best = {"f1": 0.0, "acc": 0.0}
    for answer in answers(golden):
        gold = normalize(answer)
        gold_text = " ".join(gold)
        # Match FlashRAG's token metric contract: boolean/no-answer labels
        # receive credit only on an exact normalized label match.
        special_answers = {"yes", "no", "noanswer"}
        special_mismatch = (
            pred_text in special_answers and pred_text != gold_text
        ) or (gold_text in special_answers and pred_text != gold_text)
        if special_mismatch:
            f1 = 0.0
        else:
            common = collections.Counter(pred) & collections.Counter(gold)
            overlap = sum(common.values())
            precision = overlap / len(pred) if pred else 0.0
            recall = overlap / len(gold) if gold else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        current = {
            "f1": f1,
            "acc": float(bool(gold_text) and gold_text in pred_text),
        }
        for name in best:
            best[name] = max(best[name], current[name])
    return best


def retrieval_recall_at_k(documents, golden, k: int) -> float:
    """Match FlashRAG's answer-hit Retrieval_Recall metric."""
    if k < 1:
        raise ValueError("k must be positive")
    normalized_answers = []
    for answer in answers(golden):
        normalized_answer = " ".join(normalize(answer))
        if normalized_answer:
            normalized_answers.append(normalized_answer)
    for document in list(documents or ())[:k]:
        if isinstance(document, dict):
            contents = document.get("contents", "")
        else:
            contents = str(document)
        normalized_document = " ".join(normalize(contents))
        if any(answer in normalized_document for answer in normalized_answers):
            return 1.0
    return 0.0


def collect_quality_observations(output_list, rows, topk: int, expected_count: int):
    """Require one complete Recall/F1/ACC observation per request."""
    if len(rows) != expected_count:
        raise RuntimeError(
            "Quality metric denominator mismatch: "
            f"expected {expected_count} rows, observed {len(rows)}"
        )
    predictions = [None] * expected_count
    retrieval_recalls = [None] * expected_count
    retrieval_id_rows = [None] * expected_count
    for request in output_list:
        request_id = int(request.id)
        if request_id < 0 or request_id >= expected_count:
            raise RuntimeError(f"Out-of-range output request ID: {request_id}")
        if predictions[request_id] is not None:
            raise RuntimeError(f"Duplicate output request ID: {request_id}")
        prediction = str(request.answer)
        documents = list(request.documents or ())
        predictions[request_id] = prediction
        retrieval_id_rows[request_id] = [
            str(document.get("id", "")) if isinstance(document, dict) else ""
            for document in documents[:topk]
        ]
        retrieval_recalls[request_id] = retrieval_recall_at_k(
            documents,
            rows[request_id].get("golden_answers", []),
            topk,
        )

    missing = [
        request_id
        for request_id, prediction in enumerate(predictions)
        if prediction is None
    ]
    if missing:
        raise RuntimeError(
            f"Missing quality observations for {len(missing)} requests: {missing[:10]}"
        )
    metrics = [
        score_prediction(prediction, row.get("golden_answers", []))
        for prediction, row in zip(predictions, rows)
    ]
    for request_id, (recall, metric) in enumerate(zip(retrieval_recalls, metrics)):
        for name, value in (
            ("retrieval_recall", recall),
            ("f1", metric.get("f1")),
            ("acc", metric.get("acc")),
        ):
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Non-finite {name} for request {request_id}: {value}"
                )
    return predictions, retrieval_recalls, retrieval_id_rows, metrics


class CommonPromptBuilder:
    """Reproduce MnemoRAG's PromptTemplate without importing its runtime."""

    system_template = CANONICAL_SYSTEM_PROMPT
    user_template = CANONICAL_USER_PROMPT

    def __init__(
        self,
        model_path: Path,
        max_model_len: int,
        max_tokens: int,
        tokenizer=None,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = tokenizer
        self.max_model_len = int(max_model_len)
        self.max_tokens = int(max_tokens)

    @staticmethod
    def format_docs(docs) -> str:
        parts = []
        for index, doc in enumerate(docs, 1):
            contents = (
                doc.get("contents", str(doc)) if isinstance(doc, dict) else str(doc)
            )
            title, separator, text = contents.partition("\n")
            if not separator:
                text = ""
            parts.append(f"Doc {index}(Title: {title}) {text}\n")
        return "".join(parts)

    def answer(
        self, question: str, docs=None, formatted_reference=None, previous_gen=None
    ) -> str:
        reference = (
            self.format_docs(docs or [])
            if formatted_reference is None
            else str(formatted_reference)
        )
        system = self.system_template.replace("{reference}", reference)
        if previous_gen:
            system += (
                "\n\nKnown information from previous iteration:\n"
                f"{previous_gen}\n"
                "Note: Do not repeat the draft answer verbatim unless no improvement is possible.\n"
            )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": self.user_template.replace("{question}", question),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def compression(self, question: str, docs) -> str:
        docs_text = "\n".join(
            f"Document {index}: "
            f"{doc.get('contents', str(doc)) if isinstance(doc, dict) else str(doc)}"
            for index, doc in enumerate(docs, 1)
        )
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Compress the retrieved documents for the given question.\n"
            "Keep every fact that may help answer the question, remove irrelevant "
            "or repeated text, and do not answer the question. If none of the "
            "documents contains useful evidence, output exactly <NO_EVIDENCE>.\n\n"
            f"Documents:\n{docs_text}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Question:\n{question}\n\nCompressed evidence:\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )


RECOMP_COMPRESSION_PROMPT_MARKERS = (
    "Compress the retrieved documents for the given question.",
    "Compressed evidence:\n",
)


def install_generation_budget_router(executor, args):
    """Route token budgets by immutable prompt identity, not mutable stage state."""
    original_add_request = executor.generator.add_request
    routing_stats = collections.Counter()

    def add_request(input_query, request_id, *positional, **params):
        prompt = str(input_query)
        is_recomp_compression = all(
            marker in prompt for marker in RECOMP_COMPRESSION_PROMPT_MARKERS
        )
        route = "recomp_compression" if is_recomp_compression else "default"
        max_tokens = (
            args.recomp_compression_max_tokens
            if is_recomp_compression
            else args.max_tokens
        )
        routing_stats[f"{route}_requests"] += 1
        routing_stats[f"{route}_max_tokens_{int(max_tokens)}"] += 1
        params["max_tokens"] = int(max_tokens)
        return original_add_request(
            input_query,
            request_id,
            *positional,
            **params,
        )

    executor.generator.add_request = add_request
    return routing_stats


def request_classes(retrieval_latencies_s: list[float]):
    from heterag.executor.request import Request

    class TimedRetrievalRequest(Request):
        def __init__(self, query, workflow):
            super().__init__(query, workflow=workflow)
            self._retrieval_started = None

        def timed_rtask(self, *positional, **keywords):
            self._retrieval_started = time.perf_counter()
            return self.get_rtasks(*positional, **keywords)

        def finish_retrieval(self):
            if self._retrieval_started is not None:
                retrieval_latencies_s.append(
                    time.perf_counter() - self._retrieval_started
                )
                self._retrieval_started = None

    class OnePassRequest(TimedRetrievalRequest):
        def __init__(self, query, builder):
            super().__init__(query, workflow="Sequential")
            self.builder = builder

        def update_stage(self, stage_input="", stage_id=-1, is_spec=False):
            if stage_id != self.exec_stage and stage_id != -1:
                return None
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                return [self.timed_rtask()]
            if self.exec_stage == 0:
                if not is_spec:
                    self.finish_retrieval()
                self.transfer_to_stageid(1, is_spec)
                self.update_documents(stage_input, is_spec)
                prompt = self.builder.answer(self.initial_query, docs=stage_input)
                return [self.get_gtasks(input_str=prompt, is_spec=is_spec)]
            if self.exec_stage == 1:
                self.answer = str(stage_input)
                self.current_query = str(stage_input)
                self.transfer_to_stageid(2, is_spec)
                return None
            raise ValueError("Invalid one-pass stage")

    class IRGRequest(TimedRetrievalRequest):
        def __init__(self, query, builder, rounds):
            super().__init__(query, workflow="IRG")
            self.builder = builder
            self.rounds = rounds
            self.completed_rounds = 0
            self.previous_gen = None

        def update_stage(self, stage_input="", stage_id=-1, is_spec=False):
            if stage_id != self.exec_stage and stage_id != -1:
                return None
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                return [self.timed_rtask(input_str=self.initial_query)]
            if self.exec_stage == 0:
                if not is_spec:
                    self.finish_retrieval()
                self.transfer_to_stageid(1, is_spec)
                self.update_documents(stage_input, is_spec)
                if self.previous_gen:
                    round_question = (
                        f"{self.initial_query}\n\nCurrent answer draft:\n"
                        f"{self.previous_gen}\n\n"
                        "Revise the draft using the newly retrieved evidence."
                    )
                else:
                    round_question = self.initial_query
                prompt = self.builder.answer(
                    round_question,
                    docs=stage_input,
                    previous_gen=self.previous_gen,
                )
                return [self.get_gtasks(input_str=prompt, is_spec=is_spec)]
            if self.exec_stage == 1:
                self.previous_gen = str(stage_input)
                self.answer = self.previous_gen
                self.completed_rounds += 1
                if self.completed_rounds >= self.rounds:
                    self.transfer_to_stageid(2, is_spec)
                    return None
                self.current_query = (
                    f"{self.initial_query}\n\nCurrent answer draft: {self.previous_gen}"
                )
                self.transfer_to_stageid(0, is_spec)
                return [self.timed_rtask(input_str=self.current_query)]
            raise ValueError("Invalid IRG stage")

    class RECOMPRequest(TimedRetrievalRequest):
        def __init__(self, query, builder):
            super().__init__(query, workflow="RECOMP")
            self.builder = builder

        def update_stage(self, stage_input="", stage_id=-1, is_spec=False):
            if stage_id != self.exec_stage and stage_id != -1:
                return None
            if self.exec_stage == -1:
                self.transfer_to_stageid(0, is_spec)
                return [self.timed_rtask()]
            if self.exec_stage == 0:
                if not is_spec:
                    self.finish_retrieval()
                self.transfer_to_stageid(1, is_spec)
                self.update_documents(stage_input, is_spec)
                prompt = self.builder.compression(self.initial_query, stage_input)
                return [self.get_gtasks(input_str=prompt, is_spec=is_spec)]
            if self.exec_stage == 1:
                compressed = str(stage_input).strip()
                reference = (
                    ""
                    if compressed in {"", "<NO_EVIDENCE>", "NO_EVIDENCE"}
                    else compressed
                )
                self.transfer_to_stageid(2, is_spec)
                prompt = self.builder.answer(
                    self.initial_query, formatted_reference=reference
                )
                return [self.get_gtasks(input_str=prompt, is_spec=is_spec)]
            if self.exec_stage == 2:
                self.answer = str(stage_input)
                self.transfer_to_stageid(3, is_spec)
                return None
            raise ValueError("Invalid RECOMP stage")

    return OnePassRequest, IRGRequest, RECOMPRequest


def run(args: argparse.Namespace, selected_cpus: list[int]) -> dict:
    sys.path.insert(0, str(HEDRA_ROOT))
    import faiss
    from heterag import Config
    from heterag.executor.executor import HeteragExecutor

    hedra_generator_module = importlib.import_module("heterag.executor.generator")
    from heterag.executor.request import Request
    from heterag.executor.sequential_executor import SequentialExecutor

    hot_cluster_artifact = None
    if args.system == "hedrarag":
        if args.hot_cluster_csv is None:
            raise ValueError(
                "HedraRAG measurements require --hot-cluster-csv; silent "
                "largest-cluster fallback is not accepted"
            )
        hot_cluster_path = args.hot_cluster_csv.resolve()
        if not hot_cluster_path.is_file():
            raise ValueError(f"missing hot-cluster CSV: {hot_cluster_path}")
        args.hot_cluster_csv = hot_cluster_path
        hot_cluster_artifact = {
            "path": str(hot_cluster_path),
            "sha256": file_sha256(hot_cluster_path),
            "selection_source": "disjoint_online_warmup_cluster_frequency",
        }
    elif args.hot_cluster_csv is not None:
        raise ValueError("--hot-cluster-csv is only valid for HedraRAG")

    arrival_spec = ArrivalTraceSpec(
        request_count=args.num_requests,
        request_rate=args.arrival_rate,
        burstiness=args.arrival_burstiness,
        random_seed=args.seed,
        normalize_duration=args.normalize_arrival_trace,
    )
    trace_sample_indices = None
    if args.arrival_trace is not None:
        arrivals, arrival_manifest = load_trace_bundle(args.arrival_trace)
        trace_sample_indices = load_trace_sample_indices(args.arrival_trace)
    rows, sample_indices, source_count = load_sample(
        DATASET_PATHS[args.dataset],
        args.num_requests,
        args.seed,
        explicit_indices=trace_sample_indices,
    )
    questions = [row["question"] for row in rows]
    warmup_rows = None
    warmup_sample_indices = []
    if args.online_warmup_indices is not None:
        explicit_warmup_indices = np.asarray(
            np.load(args.online_warmup_indices, allow_pickle=False),
            dtype="<i8",
        )
        warmup_rows, warmup_sample_indices, warmup_source_count = load_sample(
            DATASET_PATHS[args.dataset],
            len(explicit_warmup_indices),
            args.seed ^ 0x5A17,
            explicit_indices=explicit_warmup_indices,
        )
        if warmup_source_count != source_count:
            raise RuntimeError("warmup and measurement dataset sizes differ")
        if set(warmup_sample_indices) & set(sample_indices):
            raise RuntimeError("warmup and measurement requests must be disjoint")
    if args.arrival_trace is not None:
        validate_trace_workload(
            arrival_manifest,
            arrival_spec,
            sample_indices=sample_indices,
            dataset_sha256=file_sha256(DATASET_PATHS[args.dataset]),
        )
        arrival_trace_source = str(args.arrival_trace.resolve())
    else:
        arrivals = build_arrival_offsets(arrival_spec)
        arrival_manifest = build_trace_manifest(
            arrival_spec,
            arrivals,
            sample_indices=sample_indices,
            dataset_sha256=file_sha256(DATASET_PATHS[args.dataset]),
        )
        arrival_trace_source = "generated"
    nprobe_minibatch = 64 if args.nprobe == 512 else 32
    config = Config(
        config_dict={
            "data_dir": args.dataset,
            "data_source": "local",
            "dataset_name": args.dataset,
            "split_name": "test",
            "index_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
            ),
            "corpus_path": str(
                ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
            ),
            "corpus_source": "local",
            "nprobe": args.nprobe,
            "retrieval_topk": args.topk,
            "model2path": {
                "e5": str(ASSET_ROOT / "models/e5-large-v2"),
                "llama3-8B-instruct": str(ASSET_ROOT / "models/llama3-8B-instruct"),
            },
            "generator_model": "llama3-8B-instruct",
            "retrieval_method": "e5",
            "gpu_id": args.gpu_id,
            "retrieval_batch_size": args.retrieval_batch_size,
            "max_retrieval_batch": args.retrieval_batch_size,
            "return_embedding": args.system == "hedrarag",
            "framework": "vllm",
            "use_fp16": True,
            "pooling_method": "mean",
            "generator_max_input_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "ignore_eos": args.ignore_eos,
            },
            "task_batch": args.serving_batch_size,
            "max_num_seqs": args.serving_batch_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "iter_num": args.irg_rounds,
            "print_log": False,
            "continuous_retrieval": args.system == "hedrarag",
            "nprobe_minibatch": (
                nprobe_minibatch if args.system == "hedrarag" else args.nprobe
            ),
            "speculative_retrieval": args.system == "hedrarag",
            "spec_method": "heteRAG",
            "spec_total_size": 8 if args.system == "hedrarag" else 0,
            "gpu_onloading": args.system == "hedrarag",
            "hot_cluster_csv": (
                str(args.hot_cluster_csv) if args.system == "hedrarag" else None
            ),
            "retrieval_gpu_cache_gib": 4.0,
            "retrieval_gpu_free_fraction": 0.8,
            "request_per_second": args.arrival_rate,
            "use_llm": True,
            "seed": args.seed,
        }
    )
    builder = CommonPromptBuilder(
        ASSET_ROOT / "models/llama3-8B-instruct",
        args.max_model_len,
        args.max_tokens,
    )
    retrieval_latencies_s: list[float] = []
    OnePassRequest, IRGRequest, RECOMPRequest = request_classes(retrieval_latencies_s)

    def make_requests(input_questions):
        request_list = []
        for question in input_questions:
            if args.workflow == "onepass":
                request_list.append(OnePassRequest(question, builder))
            elif args.workflow == "irg":
                request_list.append(IRGRequest(question, builder, args.irg_rounds))
            else:
                request_list.append(RECOMPRequest(question, builder))
        return request_list

    def execute_arrivals(input_requests, input_arrivals):
        origin = time.time()
        current = 0
        while executor.execute() < len(input_requests):
            now = time.time() - origin
            due = []
            while current < len(input_requests) and now >= input_arrivals[current]:
                due.append(input_requests[current])
                current += 1
            if due:
                executor.add_requests(due)
            elif current < len(input_requests):
                time.sleep(min(0.001, max(0.0, input_arrivals[current] - now)))
        return origin, time.time() - origin

    engine_arg_calls = install_hedra_engine_args_contract(hedra_generator_module, args)
    executor = (
        HeteragExecutor(config)
        if args.system == "hedrarag"
        else SequentialExecutor(config)
    )
    generation_token_records = []
    generation_stage_occurrences = collections.Counter()
    original_generator_step = executor.generator.step

    def traced_generator_step():
        outputs = original_generator_step()
        for output in outputs:
            if not getattr(output, "finished", False):
                continue
            request_id = str(output.request_id)
            stage_ordinal = int(generation_stage_occurrences[request_id])
            generation_stage_occurrences[request_id] += 1
            candidates = list(getattr(output, "outputs", ()) or ())
            generation_token_records.append(
                {
                    "request_id": request_id,
                    "stage_ordinal": stage_ordinal,
                    "prompt_tokens": len(
                        getattr(output, "prompt_token_ids", None) or ()
                    ),
                    "completion_tokens": sum(
                        len(getattr(candidate, "token_ids", None) or ())
                        for candidate in candidates
                    ),
                    "finish_reasons": [
                        getattr(candidate, "finish_reason", None)
                        for candidate in candidates
                    ],
                    "stop_reasons": [
                        getattr(candidate, "stop_reason", None)
                        for candidate in candidates
                    ],
                }
            )
        return outputs

    executor.generator.step = traced_generator_step
    if len(engine_arg_calls) != 1:
        raise RuntimeError(
            f"expected one Hedra EngineArgs construction, observed {len(engine_arg_calls)}"
        )
    expected_vllm_contract = {
        "max_num_seqs": int(args.serving_batch_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_prefix_caching": True,
    }
    if args.max_num_batched_tokens > 0:
        expected_vllm_contract["max_num_batched_tokens"] = int(
            args.max_num_batched_tokens
        )
    vllm_engine_contract = validate_effective_vllm_contract(
        executor.generator.model,
        expected_vllm_contract,
    )
    generation_budget_routing = install_generation_budget_router(executor, args)
    online_warmup = None
    if warmup_rows is not None:
        warmup_spec = ArrivalTraceSpec(
            request_count=len(warmup_rows),
            request_rate=args.arrival_rate,
            burstiness=args.arrival_burstiness,
            random_seed=args.seed ^ 0x5A17,
            normalize_duration=args.normalize_arrival_trace,
        )
        warmup_arrivals = build_arrival_offsets(warmup_spec)
        Request.request_num = 0
        warmup_requests = make_requests([row["question"] for row in warmup_rows])
        _, warmup_elapsed_s = execute_arrivals(warmup_requests, warmup_arrivals)
        if len(executor.output_list) != len(warmup_rows):
            raise RuntimeError("online warmup did not complete every request")
        executor.reinit(config)
        retrieval_latencies_s.clear()
        generation_token_records.clear()
        generation_stage_occurrences.clear()
        generation_budget_routing.clear()
        online_warmup = {
            "status": "complete",
            "request_count": len(warmup_rows),
            "elapsed_s": warmup_elapsed_s,
            "arrival_rate_rps": float(args.arrival_rate),
            "workflow": str(args.workflow),
            "indices_path": str(args.online_warmup_indices.resolve()),
            "indices_sha256": file_sha256(args.online_warmup_indices),
            "measurement_excludes_warmup": True,
            "same_engine_instance_as_measurement": True,
            "cache_state_preserved": True,
        }

    Request.request_num = 0
    request_list = make_requests(questions)
    try:
        origin, wall_time = execute_arrivals(request_list, arrivals)
    finally:
        executor.finalize()

    finish = np.zeros(args.num_requests, dtype=float)
    for request_id, completed_at in executor.request_time_dict.items():
        if 0 <= int(request_id) < args.num_requests:
            finish[int(request_id)] = float(completed_at) - origin
    if np.any(finish <= 0):
        raise RuntimeError(
            f"Missing completion timestamps for {int(np.sum(finish <= 0))} requests"
        )
    latency = finish - arrivals
    slo_summary = summarize_latency_slo(
        latency,
        wall_time_s=wall_time,
        slo_seconds=args.latency_slo_seconds,
    )
    runtime_monitor = getattr(args, "runtime_resource_monitor", None)
    runtime_resource_summary = None
    if runtime_monitor is not None:
        runtime_monitor.checkpoint("workload:complete")
        runtime_monitor.stop()
        runtime_monitor.assert_clean()
        runtime_resource_summary = runtime_monitor.summary()
    predictions, retrieval_recalls, retrieval_id_rows, metrics = (
        collect_quality_observations(
            executor.output_list,
            rows,
            args.topk,
            args.num_requests,
        )
    )
    metric_summary = {
        name: statistics.fmean(item[name] for item in metrics) for name in ("f1", "acc")
    }
    metric_summary[f"retrieval_recall_top{args.topk}"] = statistics.fmean(
        retrieval_recalls
    )
    stage = np.asarray(executor.request_latency, dtype=float)
    stage_summary = {
        "retrieval_request_count": int(len(retrieval_latencies_s)),
        "retrieval_mean_s": (
            float(np.mean(retrieval_latencies_s)) if retrieval_latencies_s else None
        ),
        "retrieval_p95_s": (
            float(np.percentile(retrieval_latencies_s, 95))
            if retrieval_latencies_s
            else None
        ),
        "generation_request_count": int(len(stage)),
        "generation_e2e_mean_s": float(np.mean(stage[:, 0])) if len(stage) else None,
        "prefill_mean_s": float(np.mean(stage[:, 1])) if len(stage) else None,
        "decode_mean_s": float(np.mean(stage[:, 2])) if len(stage) else None,
    }
    token_records_by_stage = collections.defaultdict(list)
    for record in generation_token_records:
        token_records_by_stage[int(record["stage_ordinal"])].append(record)
    stage_summary["generation_token_accounting"] = {
        "request_count": len(generation_token_records),
        "prompt_tokens_mean": (
            float(
                np.mean(
                    [record["prompt_tokens"] for record in generation_token_records]
                )
            )
            if generation_token_records
            else None
        ),
        "completion_tokens_mean": (
            float(
                np.mean(
                    [record["completion_tokens"] for record in generation_token_records]
                )
            )
            if generation_token_records
            else None
        ),
        "by_stage_ordinal": {
            str(stage_ordinal): {
                "request_count": len(records),
                "prompt_tokens_mean": float(
                    np.mean([record["prompt_tokens"] for record in records])
                ),
                "completion_tokens_mean": float(
                    np.mean([record["completion_tokens"] for record in records])
                ),
            }
            for stage_ordinal, records in sorted(token_records_by_stage.items())
        },
    }
    stage_summary["generation_budget_routing"] = {
        "classifier": "immutable_prompt_signature",
        "counts": {
            key: int(value) for key, value in sorted(generation_budget_routing.items())
        },
    }
    point_name = (
        f"{args.system}_{args.dataset}_{args.workflow}_k{args.topk}_"
        f"np{args.nprobe}_rps{args.arrival_rate:g}_n{args.num_requests}"
    )
    point_dir = args.output_dir / point_name
    point_dir.mkdir(parents=True, exist_ok=True)
    generation_tokens_path = point_dir / "generation_token_records.jsonl"
    generation_tokens_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in generation_token_records
        ),
        encoding="utf-8",
    )
    request_metrics_path = point_dir / "request_metrics.npz"
    np.savez_compressed(
        request_metrics_path,
        sample_indices=np.asarray(sample_indices),
        arrival_offset_s=arrivals,
        finish_offset_s=finish,
        e2e_latency_s=latency,
        retrieval_recall_at_k=np.asarray(retrieval_recalls, dtype=np.float32),
        f1=np.asarray([item["f1"] for item in metrics], dtype=np.float32),
        acc=np.asarray([item["acc"] for item in metrics], dtype=np.float32),
        retrieval_ids=np.asarray(retrieval_id_rows, dtype=np.int64),
    )
    summary = {
        "status": "complete",
        "point_name": point_name,
        "system": args.system,
        "dataset": args.dataset,
        "workflow": args.workflow,
        "dataset_path": str(DATASET_PATHS[args.dataset]),
        "source_queries": source_count,
        "sample_indices": sample_indices,
        "online_warmup": online_warmup,
        "online_warmup_sample_indices": warmup_sample_indices,
        "seed": args.seed,
        "num_requests": args.num_requests,
        "arrival_rate_rps": args.arrival_rate,
        "arrival_burstiness": args.arrival_burstiness,
        "normalize_arrival_trace": args.normalize_arrival_trace,
        "arrival_trace_source": arrival_trace_source,
        "arrival_trace_manifest": arrival_manifest,
        "arrival_trace_sha256": arrival_manifest["arrival_trace_sha256"],
        "latency_slo_seconds": args.latency_slo_seconds,
        "campaign_group_id": args.campaign_group_id,
        "campaign_repeat": args.campaign_repeat,
        "campaign_order_position": args.campaign_order_position,
        "topk": args.topk,
        "nprobe": args.nprobe,
        "retrieval_batch_size": args.retrieval_batch_size,
        "serving_batch_size": args.serving_batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens or None,
        "vllm_engine_effective": vllm_engine_contract["values"],
        "vllm_engine_contract_audit": vllm_engine_contract,
        "vllm_engine_args": engine_arg_calls[0],
        "irg_rounds": args.irg_rounds,
        "nprobe_minibatch": (
            nprobe_minibatch if args.system == "hedrarag" else args.nprobe
        ),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": args.ignore_eos,
        "recomp_compression_max_tokens": args.recomp_compression_max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu_memory_policy": args.gpu_memory_policy,
        "retrieval_gpu_cache": (
            {
                **hot_cluster_artifact,
                "configured_gib": 4.0,
                "runtime": executor.retrieval_runtime_stats,
            }
            if hot_cluster_artifact is not None
            else None
        ),
        "prefix_caching": True,
        "model_path": str(ASSET_ROOT / "models/llama3-8B-instruct"),
        "retrieval_model_path": str(ASSET_ROOT / "models/e5-large-v2"),
        "index_path": str(ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"),
        "corpus_path": str(ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"),
        "physical_gpu_id": args.gpu_id,
        "visible_gpu_count": 1,
        "faiss_version": faiss.__version__,
        "faiss_path": str(Path(faiss.__file__).resolve()),
        "faiss_custom_extensions": {
            "search_preassigned_heterag": hasattr(
                faiss.IndexIVF, "search_preassigned_heterag"
            ),
            "get_cluster_size_heterag": hasattr(
                faiss.IndexIVF, "get_cluster_size_heterag"
            ),
        },
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "cpu_topology": args.cpu_topology,
        "numa_policy": args.numa_policy,
        "resource_preflight": args.resource_preflight,
        "runtime_resource_monitor": runtime_resource_summary,
        "request_metrics_artifact": {
            "path": str(request_metrics_path.resolve()),
            "sha256": file_sha256(request_metrics_path),
            "request_count": args.num_requests,
        },
        "generation_token_records_artifact": {
            "path": str(generation_tokens_path.resolve()),
            "sha256": file_sha256(generation_tokens_path),
            "request_count": len(generation_token_records),
        },
        "wall_time_s": wall_time,
        "throughput_qps": args.num_requests / wall_time,
        "request_latency_mean_s": float(np.mean(latency)),
        "request_latency_p50_s": float(np.percentile(latency, 50)),
        "request_latency_p95_s": float(np.percentile(latency, 95)),
        "request_latency_p99_s": float(np.percentile(latency, 99)),
        "latency_accounting": {
            "canonical_origin": "scheduled_arrival",
            "completion": "per_request_final_generation",
            "includes_arrival_schedule_lag": True,
        },
        **slo_summary,
        "metrics": metric_summary,
        "metric_observations": {name: args.num_requests for name in metric_summary},
        "retrieval_ids_sha256": canonical_sha256(retrieval_id_rows),
        "prediction_quality_sha256": canonical_sha256(
            [
                {"pred": prediction, "metrics": metric}
                for prediction, metric in zip(predictions, metrics)
            ]
        ),
        "metric_contract": {
            "version": 2,
            "retrieval": {
                "name": "answer_hit_recall_at_k",
                "k": args.topk,
                "scope": "final_retrieval_round",
                "denominator": "all_requests",
            },
            "generation": ["f1", "acc"],
        },
        "prompt_contract": expected_prompt_contract(
            max_model_len=args.max_model_len,
            answer_max_tokens=args.max_tokens,
            irg_round_max_tokens=args.max_tokens,
            recomp_compression_max_tokens=args.recomp_compression_max_tokens,
            ordering_mode=BASELINE_PROMPT_ORDERING_MODES[args.system],
        ),
        "stage_latency": stage_summary,
        "executor_cpu_time_s": float(executor.cpu_time),
        "executor_gpu_time_s": float(executor.gpu_time),
        "prompt_and_workflow_contract": "byte-matched MnemoRAG v3",
        "implementation_fidelity": implementation_fidelity(
            args.system,
            topk=args.topk,
            nprobe=args.nprobe,
        ),
        "scheduler_and_retriever": "see implementation_fidelity",
        "cache_isolation": "fresh process and fresh vLLM engine per point",
        "versions": runtime_versions(),
    }
    temporary = point_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(point_dir / "summary.json")
    print(
        "BASELINE_POINT_COMPLETE "
        + json.dumps(
            {
                "point_name": point_name,
                "wall_time_s": wall_time,
                "request_latency_mean_s": summary["request_latency_mean_s"],
                "request_latency_p95_s": summary["request_latency_p95_s"],
                "metrics": metric_summary,
                "stage_latency": stage_summary,
            }
        ),
        flush=True,
    )
    return summary


def main() -> None:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = parse_args()
    ensure_native_hedra_faiss(args)
    selected = configure(args)
    monitor = RuntimeResourceMonitor(
        selected,
        args.gpu_id,
        require_clean=args.require_clean_resources,
        max_external_cpu_busy_percent=args.max_runtime_external_cpu_busy_percent,
        sample_interval_s=args.runtime_resource_sample_seconds,
    )
    args.runtime_resource_monitor = monitor
    monitor.start()
    try:
        run(args, selected)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
