#!/usr/bin/env python3
"""Run one fair six-system retrieval point on a locked 512-query trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
HEDRA_ROOT = Path(
    os.environ.get("HEDRARAG_ROOT", PROJECT_ROOT / "third_party/HedraRAG")
).resolve()
HEDRA_FAISS_ROOT = Path(
    os.environ.get("HEDRARAG_FAISS_ROOT", HEDRA_ROOT / "third_party/faiss")
).resolve()
HEDRA_FAISS_PYTHON = HEDRA_FAISS_ROOT / "python"
HEDRA_FAISS_LIBRARY = HEDRA_FAISS_ROOT
GPTCACHE_PYTHON = Path(
    os.environ.get("GPTCACHE_PYTHON", PROJECT_ROOT / "third_party/gptcache/python")
).resolve()
INDEX_PATH = ASSET_ROOT / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"
CORPUS_PATH = ASSET_ROOT / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"
MODEL_PATH = ASSET_ROOT / "models/e5-large-v2"
DATASET_PATHS = {
    "squad": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
    "nq": ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    "hotpotqa": ASSET_ROOT / "qa/datasets/hotpotqa/train.jsonl",
    "2wikimultihopqa": ASSET_ROOT / "qa/datasets/2wikimultihopqa/train.jsonl",
}
SYSTEMS = (
    "gptcache",
    "qvcache",
    "has",
    "proximity",
    "hedrarag",
    "mnemorag",
    "faiss_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_PATHS), required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--history-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--gptcache-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--proximity-tau", type=float, default=0.48)
    parser.add_argument("--has-threshold", type=float, default=0.6)
    parser.add_argument("--quality-reference-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_hedra_environment(args: argparse.Namespace) -> None:
    if args.system != "hedrarag" or os.environ.get("HEDRA_CUSTOM_FAISS_ACTIVE") == "1":
        return
    env = os.environ.copy()
    env["HEDRA_CUSTOM_FAISS_ACTIVE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HEDRA_FAISS_PYTHON), str(HEDRA_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(HEDRA_FAISS_LIBRARY), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def configure(args: argparse.Namespace) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads != 64 or len(available) < args.cpu_threads:
        raise ValueError("This comparison requires exactly 64 available physical CPUs")
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    if selected != list(range(64)):
        raise ValueError(f"Expected CPU 0-63, got {selected}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MNEMORAG_FAISS_OMP_THREADS"] = "1"
    os.environ["MNEMORAG_FAISS_SEARCH_WORKERS"] = "8"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    return selected


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_workload(args: argparse.Namespace) -> tuple[list[str], np.ndarray, dict]:
    with np.load(args.workload, allow_pickle=False) as workload:
        indices = np.asarray(workload["sample_indices"], dtype=np.int64)
        arrivals = np.asarray(workload["arrival_offsets"], dtype=np.float64)
        manifest = json.loads(str(workload["manifest_json"].item()))
    rows = load_rows(DATASET_PATHS[args.dataset])
    if len(indices) != 512 or len(arrivals) != 512:
        raise ValueError("Locked comparison requires 512 requests")
    if np.any(indices < 0) or np.any(indices >= len(rows)):
        raise ValueError("Workload sample index is outside the dataset")
    return [str(rows[index]["question"]) for index in indices], arrivals, manifest


def load_history(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as snapshot:
        history = {name: np.asarray(snapshot[name]) for name in snapshot.files}
    metadata = json.loads(str(history["metadata_json"].item()))
    if metadata.get("format") != "mnemorag.query-base.v1":
        raise ValueError("Unsupported history snapshot")
    if history["embeddings"].ndim != 2 or history["embeddings"].shape[1] != 1024:
        raise ValueError(f"Expected Nx1024 history, got {history['embeddings'].shape}")
    if not 2_000 <= history["embeddings"].shape[0] <= 2_048:
        raise ValueError(f"Unexpected history size: {history['embeddings'].shape[0]}")
    if history["topk_doc_ids"].shape[1] < 3:
        raise ValueError("History snapshot does not contain top-k=3 results")
    return history


def direct_args(args: argparse.Namespace, system: str) -> argparse.Namespace:
    return argparse.Namespace(
        system=system,
        nprobe=args.nprobe,
        topk=args.topk,
        gpu_id=args.gpu_id,
        cpu_threads=args.cpu_threads,
        retrieval_batch_size=args.max_batch_size,
        cache_capacity=80_000,
        similarity_t_reuse=0.95,
        similarity_t_guided=0.70,
        exact_query_cache=True,
        has_cache_capacity=2_048,
        proximity_tau=args.proximity_tau,
        has_threshold=args.has_threshold,
        qvcache_regional_theta=True,
        qvcache_lazy_theta_updates=True,
        qvcache_metric="cosine",
        direct_faiss_mode="query_parallel",
        effective_faiss_omp_threads=1,
        effective_faiss_search_workers=8,
        seed=20260923,
    )


class GPTCacheRetrievalEngine:
    """Use GPTCache's official SQLite+FAISS manager as a retrieval-result cache."""

    def __init__(self, args: argparse.Namespace, point_dir: Path) -> None:
        import faiss

        from mnemorag.retriever.encoder import Encoder
        from mnemorag.retriever.utils import load_corpus

        sys.path.append(str(GPTCACHE_PYTHON))
        from gptcache.manager import CacheBase, VectorBase, get_data_manager

        self.args = args
        self.faiss = faiss
        self.encoder = Encoder(
            model_name="e5",
            model_path=str(MODEL_PATH),
            pooling_method="mean",
            max_length=128,
            use_fp16=True,
            instruction=None,
            silent=True,
        )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.index.nprobe = args.nprobe
        faiss.omp_set_num_threads(8)
        self.corpus = load_corpus(str(CORPUS_PATH))
        cache_dir = point_dir / "gptcache_store"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_manager = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{cache_dir / 'cache.db'}"),
            VectorBase(
                "faiss",
                dimension=1024,
                index_path=str(cache_dir / "vectors.index"),
            ),
            max_size=80_000,
        )
        self.distance_threshold = 2.0 - 2.0 * args.gptcache_similarity_threshold
        self.hits = 0
        self.misses = 0

    def warm(self, history: dict[str, np.ndarray]) -> None:
        embeddings = np.asarray(history["embeddings"], dtype=np.float32)
        doc_ids = np.asarray(
            history["topk_doc_ids"][:, : self.args.topk], dtype=np.int64
        )
        exact_values = np.asarray(history["exact_query_values"], dtype=np.str_)
        exact_present = np.asarray(history["exact_query_present"], dtype=np.bool_)
        for index, (embedding, ids) in enumerate(zip(embeddings, doc_ids)):
            question = (
                str(exact_values[index]) if exact_present[index] else f"history-{index}"
            )
            self.data_manager.save(question, json.dumps(ids.tolist()), embedding)
        self.data_manager.flush()

    def search(self, questions: list[str]) -> tuple[np.ndarray, dict]:
        embeddings = np.asarray(
            self.encoder.encode(
                questions, batch_size=self.args.max_batch_size, is_query=True
            ),
            dtype=np.float32,
        )
        returned = np.empty((len(questions), self.args.topk), dtype=np.int64)
        for offset, (question, embedding) in enumerate(zip(questions, embeddings)):
            matches = self.data_manager.search(embedding, top_k=1)
            if matches and float(matches[0][0]) <= self.distance_threshold:
                record = self.data_manager.get_scalar_data(matches[0])
                returned[offset] = np.asarray(
                    json.loads(record.answers[0].answer), dtype=np.int64
                )[: self.args.topk]
                self.hits += 1
            else:
                params = self.faiss.SearchParametersIVF()
                params.nprobe = self.args.nprobe
                _, ids = self.index.search(
                    embedding.reshape(1, -1), self.args.topk, params=params
                )
                returned[offset] = ids[0]
                self.data_manager.save(question, json.dumps(ids[0].tolist()), embedding)
                self.misses += 1
        _ = [self.corpus[int(index)] for index in returned.reshape(-1)]
        return returned, {"cache_hits": self.hits, "cache_misses": self.misses}

    def close(self) -> None:
        self.data_manager.flush()
        self.data_manager.close()


def initialize_standard_engine(
    args: argparse.Namespace, point_dir: Path, history: dict[str, np.ndarray]
):
    import retrieval_cache_point as cache_point

    if args.system == "gptcache":
        engine = GPTCacheRetrievalEngine(args, point_dir)
        engine.warm(history)
        return engine, {"history_load_mode": "gptcache_official_import"}

    engine_system = "faiss" if args.system == "faiss_reference" else args.system
    engine_args = direct_args(args, engine_system)
    engine = (
        cache_point.MnemoEngine(engine_args)
        if engine_system == "mnemorag"
        else cache_point.DirectEngine(engine_args)
    )
    embeddings = np.asarray(history["embeddings"], dtype=np.float32)
    ids = np.asarray(history["topk_doc_ids"][:, : args.topk], dtype=np.int64)
    scores = np.asarray(history["topk_scores"][:, : args.topk], dtype=np.float32)

    started = time.perf_counter()
    load_mode = "none"
    if engine_system == "mnemorag":
        metadata = engine.query_pool.load_snapshot(args.history_snapshot)
        load_mode = "mnemorag_query_pool_snapshot"
        if int(metadata["entry_count"]) != len(embeddings):
            raise ValueError("MnemoRAG did not load all history entries")
    elif engine_system == "proximity":
        for embedding, row_ids, row_scores in zip(embeddings, ids, scores):
            engine.cache.insert(
                embedding.tolist(),
                (row_ids.tolist(), row_scores.tolist()),
                args.proximity_tau,
            )
        load_mode = "proximity_fifo_preload"
    elif engine_system == "has":
        for index, (embedding, row_ids) in enumerate(zip(embeddings, ids)):
            engine.has_cache.add(
                {"id": index},
                row_ids,
                engine.vector_store.fetch_array(row_ids),
            )
        engine.has_query_id = len(embeddings)
        load_mode = "has_cachechannel_preload"
    elif engine_system == "qvcache":
        for start in range(0, len(embeddings), 64):
            engine._search_qvcache(embeddings[start : start + 64])
        engine.cache_hits = 0
        engine.cache_misses = 0
        load_mode = "qvcache_native_warm_search"
    return engine, {
        "history_load_mode": load_mode,
        "history_load_s": time.perf_counter() - started,
    }


def run_standard(
    args: argparse.Namespace,
    questions: list[str],
    arrivals: np.ndarray,
    point_dir: Path,
    history: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    engine, warmup = initialize_standard_engine(args, point_dir, history)
    returned = np.full((len(questions), args.topk), -1, dtype=np.int64)
    finish = np.zeros(len(questions), dtype=np.float64)
    service_times = []
    batch_sizes = []
    last_stats = {}
    try:
        if args.quality_reference_only:
            for start in range(0, len(questions), args.max_batch_size):
                ids, last_stats = engine.search(
                    questions[start : start + args.max_batch_size]
                )
                returned[start : start + len(ids)] = ids
            return finish, returned, last_stats, warmup

        origin = time.perf_counter()
        current = 0
        pending: list[int] = []
        while current < len(questions) or pending:
            elapsed = time.perf_counter() - origin
            while current < len(questions) and arrivals[current] <= elapsed:
                pending.append(current)
                current += 1
            if not pending:
                delay = arrivals[current] - elapsed
                time.sleep(min(0.001, max(0.0, float(delay))))
                continue
            batch = pending[: args.max_batch_size]
            del pending[: len(batch)]
            started = time.perf_counter()
            ids, last_stats = engine.search([questions[index] for index in batch])
            service_times.append(time.perf_counter() - started)
            batch_sizes.append(len(batch))
            completed = time.perf_counter() - origin
            for local, index in enumerate(batch):
                returned[index] = ids[local]
                finish[index] = completed
        warmup["wall_time_s"] = time.perf_counter() - origin
    finally:
        engine.close()
    last_stats["service_batch_time_mean_s"] = statistics.fmean(service_times)
    last_stats["batch_size_mean"] = statistics.fmean(batch_sizes)
    last_stats["batch_size_max"] = max(batch_sizes)
    return finish, returned, last_stats, warmup


def run_hedra(
    args: argparse.Namespace,
    questions: list[str],
    arrivals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    sys.path.insert(0, str(HEDRA_ROOT))
    from heterag import Config
    from heterag.retriever import hete_engine_spec_onloading as hedra_module
    from heterag.utils import TaskID

    nprobe_minibatch = 32
    config = Config(
        config_dict={
            "data_dir": args.dataset,
            "data_source": "local",
            "dataset_name": args.dataset,
            "split_name": "test",
            "index_path": str(INDEX_PATH),
            "corpus_path": str(CORPUS_PATH),
            "corpus_source": "local",
            "nprobe": args.nprobe,
            "retrieval_topk": args.topk,
            "model2path": {"e5": str(MODEL_PATH)},
            "generator_model": "e5",
            "retrieval_method": "e5",
            "gpu_id": args.gpu_id,
            "retrieval_batch_size": 256,
            "max_retrieval_batch": 64,
            "return_embedding": True,
            "use_fp16": True,
            "pooling_method": "mean",
            "continuous_retrieval": True,
            "nprobe_minibatch": nprobe_minibatch,
            "speculative_retrieval": True,
            "spec_method": "heteRAG",
            "spec_total_size": 8,
            "gpu_onloading": True,
            "retrieval_gpu_cache_gib": 4.0,
            "retrieval_gpu_free_fraction": 0.8,
            "use_llm": False,
            "seed": 20260923,
        }
    )
    captured: list[np.ndarray] = []
    original_load_docs = hedra_module.load_docs

    def traced_load_docs(corpus, doc_idxs):
        captured.append(np.asarray(doc_idxs, dtype=np.int64).copy())
        return original_load_docs(corpus, doc_idxs)

    hedra_module.load_docs = traced_load_docs
    engine = hedra_module.DenseHeteEngineSpecOnloading(config)
    finish = np.zeros(len(questions), dtype=np.float64)
    returned = np.full((len(questions), args.topk), -1, dtype=np.int64)
    origin = time.perf_counter()
    current = 0
    completed = set()
    while len(completed) < len(questions):
        elapsed = time.perf_counter() - origin
        due = []
        while current < len(questions) and arrivals[current] <= elapsed:
            due.append(current)
            current += 1
        if due:
            engine.add_requests(
                [questions[index] for index in due],
                [TaskID(index, 0, "R", questions[index]) for index in due],
                [args.nprobe for _ in due],
            )
        if not engine.request_dict and not engine.new_request_list:
            if current < len(questions):
                time.sleep(min(0.001, max(0.0, float(arrivals[current] - elapsed))))
                continue
            missing = sorted(set(range(len(questions))) - completed)
            raise RuntimeError(
                f"HedraRAG drained its engine without reporting {len(missing)} "
                f"final completions; first missing IDs: {missing[:8]}"
            )
        captured.clear()
        task_ids, documents = engine.step()
        flat_ids = captured[-1] if captured else np.empty((0,), dtype=np.int64)
        if len(task_ids) and len(flat_ids) != len(task_ids) * args.topk:
            raise RuntimeError("HedraRAG result-ID capture length mismatch")
        ids = flat_ids.reshape(len(task_ids), args.topk) if len(task_ids) else []
        now = time.perf_counter() - origin
        for task, row_ids in zip(task_ids, ids):
            request_id = int(task.id)
            # HedraRAG reuses the TaskID object after speculative emission and
            # does not clear its is_spec flag.  Removal from request_dict is the
            # authoritative final-completion signal in the official engine.
            if request_id in engine.request_dict:
                continue
            if request_id not in completed:
                completed.add(request_id)
                returned[request_id] = row_ids
                finish[request_id] = now
    stats = engine.runtime_stats()
    stats["wall_time_s"] = time.perf_counter() - origin
    return (
        finish,
        returned,
        stats,
        {"history_load_mode": "native_fixed_hot_cluster_cache"},
    )


def main() -> None:
    args = parse_args()
    ensure_hedra_environment(args)
    selected_cpus = configure(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_dir = args.output_dir / args.dataset / args.system
    point_dir.mkdir(parents=True, exist_ok=True)
    questions, arrivals, workload_manifest = load_workload(args)
    history = load_history(args.history_snapshot)

    if args.system == "hedrarag":
        finish, returned, stats, warmup = run_hedra(args, questions, arrivals)
    else:
        finish, returned, stats, warmup = run_standard(
            args, questions, arrivals, point_dir, history
        )

    latency = finish - arrivals
    if args.quality_reference_only:
        latency = np.asarray([], dtype=np.float64)
    np.savez_compressed(
        point_dir / "requests.npz",
        arrival_offset_s=arrivals,
        finish_offset_s=finish,
        retrieval_latency_s=latency,
        returned_doc_ids=returned,
    )
    summary = {
        "status": "complete",
        "system": args.system,
        "dataset": args.dataset,
        "metric": "request-level retrieval latency under the locked Poisson trace",
        "quality_reference_only": args.quality_reference_only,
        "num_requests": len(questions),
        "arrival_rate_qps": float(workload_manifest["realized_request_rate"]),
        "nprobe": args.nprobe,
        "topk": args.topk,
        "max_batch_size": args.max_batch_size,
        "batch_policy": "drain_currently_arrived_without_waiting_for_future_requests",
        "physical_gpu_id": int(args.gpu_id),
        "cpu_affinity": selected_cpus,
        "numa_nodes": [0, 1, 2, 3],
        "workload_path": str(args.workload.resolve()),
        "workload_sha256": sha256(args.workload),
        "workload_manifest": workload_manifest,
        "history_snapshot_path": str(args.history_snapshot.resolve()),
        "history_snapshot_sha256": sha256(args.history_snapshot),
        "history_entries": int(history["embeddings"].shape[0]),
        "warmup": warmup,
        "system_stats": stats,
        "latency_mean_s": float(np.mean(latency)) if len(latency) else None,
        "latency_p50_s": float(np.percentile(latency, 50)) if len(latency) else None,
        "latency_p95_s": float(np.percentile(latency, 95)) if len(latency) else None,
        "latency_p99_s": float(np.percentile(latency, 99)) if len(latency) else None,
        "gptcache_similarity_threshold": (
            args.gptcache_similarity_threshold if args.system == "gptcache" else None
        ),
        "proximity_tau": args.proximity_tau if args.system == "proximity" else None,
        "has_threshold": args.has_threshold if args.system == "has" else None,
        "implementation": {
            "gptcache": "official GPTCache 0.1.44 SQLite+FAISS semantic cache",
            "qvcache": "official QVCache native extension with shared IVF backend",
            "has": "official HaS CacheChannel with shared IVF fallback",
            "proximity": "official Proximity native FIFO cache with shared IVF fallback",
            "hedrarag": "official AE custom FAISS heterogeneous CPU/GPU retriever",
            "mnemorag": "MnemoRAG query base plus similarity-aware retrieval",
            "faiss_reference": "shared IVF4096,Flat quality reference",
        }[args.system],
    }
    (point_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "system": args.system,
                "dataset": args.dataset,
                "latency_mean_s": summary["latency_mean_s"],
                "stats": stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
