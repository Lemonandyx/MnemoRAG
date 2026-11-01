#!/usr/bin/env python3
"""Reproduce MnemoRAG Figure 2(a) on one dataset per invocation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("MNEMORAG_ASSET_ROOT", PROJECT_ROOT / "assets")
).resolve()
DATASET_PATHS = {
    "nq": ASSET_ROOT / "qa/datasets/nq/train.jsonl",
    "hotpotqa": ASSET_ROOT / "qa/datasets/hotpotqa/train.jsonl",
    "squad": ASSET_ROOT / "qa/datasets/squad/squad_dev.jsonl",
    "2wikimultihopqa": ASSET_ROOT / "qa/datasets/2wikimultihopqa/train.jsonl",
}
TOPK_VALUES = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PATHS))
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--cpu-threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/background_motivation",
    )
    return parser.parse_args()


def configure_resources(args: argparse.Namespace) -> list[int]:
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    gpu_id = str(args.gpu_id).strip()
    if not gpu_id or "," in gpu_id:
        raise ValueError("--gpu-id must select exactly one GPU")
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_threads > len(available):
        raise ValueError(
            f"Requested {args.cpu_threads} CPU threads but only {len(available)} are available"
        )
    selected = available[: args.cpu_threads]
    os.sched_setaffinity(0, set(selected))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    for name in (
        "MNEMORAG_FAISS_OMP_THREADS",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_PERSISTENT_ASYNC_ENGINE"] = "1"
    os.environ["VLLM_KV_FREE_BLOCK_POLICY"] = "heat"
    os.environ["VLLM_MAX_NUM_SEQS"] = "8"
    return selected


def load_sample(path: Path, count: int, seed: int):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if count > len(rows):
        raise ValueError(
            f"Requested {count} rows from a dataset containing {len(rows)}"
        )
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[index] for index in indices], indices, len(rows)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def timing_value(metrics, *names: str) -> float | None:
    """Read a timing field across the vLLM v0/v1 metrics schemas."""
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            return float(value)
    return None


def fit_prompt_to_context(prompt: str, tokenizer, max_prompt_tokens: int):
    """Keep the prompt header and final question when a retrieved context is long."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_tokens = len(token_ids)
    if original_tokens <= max_prompt_tokens:
        return prompt, original_tokens, original_tokens, False
    left = max_prompt_tokens // 2
    right = max_prompt_tokens - left
    fitted_ids = token_ids[:left] + token_ids[-right:]
    fitted_prompt = tokenizer.decode(fitted_ids, skip_special_tokens=False)
    return fitted_prompt, original_tokens, len(fitted_ids), True


async def run(args: argparse.Namespace, selected_cpus: list[int]) -> None:
    from mnemorag.config import Config
    from mnemorag.pipeline import AsyncPipeline
    from mnemorag.prompt import PromptTemplate

    source_path = DATASET_PATHS[args.dataset]
    rows, sample_indices, source_count = load_sample(
        source_path, args.num_requests, args.seed
    )
    questions = [row["question"] for row in rows]
    root = ASSET_ROOT
    config = Config(
        config_dict={
            "split": ["test"],
            "dataset_name": args.dataset,
            "data_dir": str(root / "qa/datasets"),
            "index_path": str(root / "qa/wiki18_2100w/indexes/e5_IVF4096,Flat.index"),
            "corpus_path": str(root / "qa/wiki18_2100w/corpus/wiki18_2100w.jsonl"),
            "model2path": {
                "e5": str(root / "models/e5-large-v2"),
                "llama3-8B-instruct": str(root / "models/llama3-8B-instruct"),
            },
            "generator_model": "llama3-8B-instruct",
            "retrieval_method": "e5",
            "framework": "vllm",
            "metrics": ["em", "f1"],
            "retrieval_topk": 10,
            "retrieval_batch_size": 256,
            "generator_batch_size": 256,
            "generator_max_input_len": 2048,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "ivf_nprobe": 128,
            # Figure 2 is a background characterization of vanilla RAG.
            "enable_similarity_aware": False,
            "enable_hotness_sort": False,
            "gpu_id": str(args.gpu_id),
            "cpu_threads": args.cpu_threads,
            "irg_max_iter": 3,
            "seed": args.seed,
            "disable_save": True,
        }
    )
    prompt_template = PromptTemplate(
        config,
        system_prompt=(
            "Answer the question based on the given document. Only give me the "
            "answer and do not output any other words.\nDocuments:\n{reference}"
        ),
        user_prompt="Question: {question}\nAnswer:",
    )
    pipeline = AsyncPipeline(config, prompt_template=prompt_template)
    if pipeline.generator.tensor_parallel_size != 1:
        raise RuntimeError(
            f"Expected tensor_parallel_size=1, got {pipeline.generator.tensor_parallel_size}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"figure2a_{args.dataset}_requests.jsonl"
    all_records: list[dict] = []
    summaries: list[dict] = []

    for topk in TOPK_VALUES:
        await pipeline.generator.model.reset_prefix_cache()
        pipeline.generator.exact_prefix_cached_tokens = 0
        pipeline.generator.exact_prefix_prompt_tokens = 0
        retrieval_values = []
        prefill_values = []
        decode_values = []
        end_to_end_values = []
        cached_token_values = []
        prompt_token_values = []
        truncated_prompt_count = 0
        run_started = time.perf_counter()

        for request_index, question in enumerate(questions):
            target_arrival = run_started + request_index / args.arrival_rate
            delay = target_arrival - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            end_to_end_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            (
                retrieval_results,
                retrieval_scores,
                retrieval_ids,
            ) = await asyncio.to_thread(
                pipeline.retriever.batch_search,
                [question],
                num=topk,
                query_pool=pipeline.query_pool,
                return_score=True,
                return_idx=True,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
            prompt = prompt_template.get_string(
                question=question,
                retrieval_result=retrieval_results[0],
            )
            prompt, original_prompt_tokens, fitted_prompt_tokens, was_truncated = (
                fit_prompt_to_context(
                    prompt,
                    pipeline.generator.tokenizer,
                    2048 - args.max_tokens - 8,
                )
            )
            truncated_prompt_count += int(was_truncated)
            outputs = await pipeline.generator.generate_async(
                [prompt], return_raw_output=True
            )
            output = outputs[0]
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError(
                    "vLLM did not return complete request timing metrics"
                )
            first_token_time = timing_value(
                metrics, "first_token_time", "first_token_ts"
            )
            last_token_time = timing_value(
                metrics, "finished_time", "last_token_time", "last_token_ts"
            )
            scheduled_time = timing_value(metrics, "scheduled_time", "scheduled_ts")
            arrival_time = timing_value(metrics, "arrival_time")
            first_token_latency = timing_value(metrics, "first_token_latency")
            if first_token_time is None or last_token_time is None:
                raise RuntimeError(
                    f"vLLM returned an unsupported metrics schema: {type(metrics)!r}"
                )

            # Figure 2(a) separates the LLM execution stages, so prefill starts
            # when the engine schedules the request.  Fall back to vLLM's TTFT
            # field (or the legacy wall-clock timestamps) for older releases.
            if scheduled_time is not None and scheduled_time > 0:
                prefill_ms = (first_token_time - scheduled_time) * 1000.0
            elif first_token_latency is not None:
                prefill_ms = first_token_latency * 1000.0
            elif arrival_time is not None:
                prefill_ms = (first_token_time - arrival_time) * 1000.0
            else:
                raise RuntimeError("vLLM did not provide a prefill timing origin")
            decode_ms = (last_token_time - first_token_time) * 1000.0
            end_to_end_ms = (time.perf_counter() - end_to_end_started) * 1000.0
            cached_tokens = int(getattr(output, "num_cached_tokens", 0) or 0)
            prompt_tokens = len(getattr(output, "prompt_token_ids", None) or ())

            record = {
                "dataset": args.dataset,
                "source_queries": source_count,
                "sample_size": args.num_requests,
                "sample_index": sample_indices[request_index],
                "request_index": request_index,
                "topk": topk,
                "nprobe": 128,
                "configured_rbs": 256,
                "arrival_rate_rps": args.arrival_rate,
                "retrieval_ms": retrieval_ms,
                "prefill_ttft_ms": prefill_ms,
                "decode_ms": decode_ms,
                "generation_ms": prefill_ms + decode_ms,
                "end_to_end_ms": end_to_end_ms,
                "cached_tokens": cached_tokens,
                "prompt_tokens": prompt_tokens,
                "original_prompt_tokens": original_prompt_tokens,
                "fitted_prompt_tokens": fitted_prompt_tokens,
                "prompt_was_truncated": was_truncated,
                "generated_tokens": len(output.outputs[0].token_ids),
                "retrieved_ids": retrieval_ids[0],
                "retrieval_scores": [float(value) for value in retrieval_scores[0]],
            }
            all_records.append(record)
            retrieval_values.append(retrieval_ms)
            prefill_values.append(prefill_ms)
            decode_values.append(decode_ms)
            end_to_end_values.append(end_to_end_ms)
            cached_token_values.append(cached_tokens)
            prompt_token_values.append(prompt_tokens)

        with raw_path.open("w", encoding="utf-8") as stream:
            for record in all_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        cached_total = sum(cached_token_values)
        prompt_total = sum(prompt_token_values)
        summaries.append(
            {
                "dataset": args.dataset,
                "num_requests": args.num_requests,
                "topk": topk,
                "nprobe": 128,
                "rbs": 256,
                "arrival_rate_rps": args.arrival_rate,
                "mean_retrieval_ms": statistics.fmean(retrieval_values),
                "stdev_retrieval_ms": statistics.stdev(retrieval_values),
                "mean_prefill_ttft_ms": statistics.fmean(prefill_values),
                "stdev_prefill_ttft_ms": statistics.stdev(prefill_values),
                "mean_decode_ms": statistics.fmean(decode_values),
                "stdev_decode_ms": statistics.stdev(decode_values),
                "mean_end_to_end_ms": statistics.fmean(end_to_end_values),
                "prefix_hit_rate": cached_total / prompt_total if prompt_total else 0.0,
                "truncated_prompts": truncated_prompt_count,
            }
        )
        write_csv(
            args.output_dir / f"figure2a_{args.dataset}_summary.csv",
            summaries,
            list(summaries[0]),
        )
        print(
            f"TOPK_DONE dataset={args.dataset} topk={topk} "
            f"retrieval_ms={summaries[-1]['mean_retrieval_ms']:.3f} "
            f"prefill_ms={summaries[-1]['mean_prefill_ttft_ms']:.3f} "
            f"decode_ms={summaries[-1]['mean_decode_ms']:.3f}",
            flush=True,
        )

    manifest = {
        "paper_figure": "2a",
        "dataset": args.dataset,
        "dataset_path": str(source_path),
        "source_queries": source_count,
        "sample_indices": sample_indices,
        "sample_size": args.num_requests,
        "seed": args.seed,
        "arrival_rate_rps": args.arrival_rate,
        "nprobe": 128,
        "rbs": 256,
        "topk_values": list(TOPK_VALUES),
        "max_tokens": args.max_tokens,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": selected_cpus,
        "physical_gpu_id": str(args.gpu_id),
        "visible_gpu_count": 1,
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_prompt_tokens": 2048 - args.max_tokens - 8,
        "prompt_truncation": "preserve_first_and_last_tokens",
        "prefix_caching": True,
        "similarity_aware_retrieval": False,
        "hotness_sort": False,
    }
    (args.output_dir / f"figure2a_{args.dataset}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pipeline.generator.model.shutdown()
    print(f"STAGE_LATENCY_OK dataset={args.dataset}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    args = parse_args()
    selected_cpus = configure_resources(args)
    asyncio.run(run(args, selected_cpus))


if __name__ == "__main__":
    main()
