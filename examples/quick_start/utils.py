"""Online request scheduler used by the formal MnemoRAG benchmarks.

The scheduler deliberately keeps arrival pacing, workflow concurrency, and
vLLM admission as separate layers. Requests enter an unbounded arrival queue
at their configured timestamps; workers drain only requests that are already
ready, and the persistent AsyncLLM engine owns generation backpressure.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from mnemorag.dataset.dataset import Dataset


def build_online_arrival_offsets(
    request_count,
    request_rate,
    *,
    burstiness=1.0,
    random_seed=0,
    normalize_duration=True,
):
    """Return finite vLLM-style arrival offsets for an open-loop workload."""
    request_count = int(request_count)
    request_rate = float(request_rate)
    burstiness = float(burstiness)
    if request_count < 0:
        raise ValueError("request_count must be nonnegative")
    if request_count == 0:
        return np.asarray([], dtype=float)
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")
    if np.isinf(request_rate):
        return np.zeros(request_count, dtype=float)
    if np.isinf(burstiness):
        intervals = np.full(request_count, 1.0 / request_rate, dtype=float)
    else:
        rng = np.random.default_rng(random_seed)
        intervals = rng.gamma(
            shape=burstiness,
            scale=1.0 / (request_rate * burstiness),
            size=request_count,
        )
    offsets = np.cumsum(intervals)
    if normalize_duration and offsets[-1] > 0:
        offsets *= (request_count / request_rate) / offsets[-1]
    return offsets


async def poisson_persistent_workflow_run(
    test_data,
    pipeline,
    arrival_rate,
    workflow,
    do_eval=True,
    worker_count=8,
    microbatch_max_size=32,
    random_seed=0,
    arrival_offsets=None,
    arrival_burstiness=1.0,
    normalize_arrival_trace=True,
    record_request_timing=False,
):
    """Serve One-Pass, IRG, or RECOMP requests on an open-loop timeline.

    Retrieval remains serialized by ``AsyncPipeline._retrieve_async``. Each
    worker returns to the request queue only after its workflow batch finishes,
    while generation submitted by another worker remains active in AsyncLLM.
    """
    workflow_name = str(workflow).strip().upper()
    recomp_runner = getattr(pipeline, "run_stream_recomp", None)
    if recomp_runner is None:
        recomp_runner = pipeline.run_stream_dc
    workflow_runners = {
        "ONEPASS": pipeline.run_stream,
        "IRG": pipeline.run_stream_irg,
        "RECOMP": recomp_runner,
    }
    if workflow_name not in workflow_runners:
        raise ValueError(
            "poisson_persistent_workflow_run supports ONEPASS, IRG, and RECOMP"
        )

    source_rows = [dict(item.data) for item in test_data]
    total_n = len(source_rows)
    if total_n == 0:
        empty = Dataset(data=[])
        pipeline.last_workflow_scheduler_stats = {
            "workflow": workflow_name,
            "requests": 0,
            "batches": 0,
            "request_batching_policy": "worker_nonblocking_drain",
            "microbatch_policy": "worker_dispatch_drain",
            "preformed_work_queue": False,
        }
        return pipeline.evaluate(empty, do_eval=do_eval)

    output_buffer = [None] * total_n
    finish_offsets = [None] * total_n
    actual_arrival_offsets = [None] * total_n
    run_origin = time.perf_counter()
    if arrival_offsets is None:
        scheduled_arrivals = build_online_arrival_offsets(
            total_n,
            arrival_rate,
            burstiness=arrival_burstiness,
            random_seed=random_seed,
            normalize_duration=normalize_arrival_trace,
        )
        arrival_trace_source = "generated"
        arrival_trace_normalized = bool(normalize_arrival_trace)
    else:
        scheduled_arrivals = np.asarray(arrival_offsets, dtype=float)
        if scheduled_arrivals.shape != (total_n,):
            raise ValueError(
                "arrival_offsets must contain exactly one value per request"
            )
        if np.any(scheduled_arrivals < 0) or np.any(
            np.diff(scheduled_arrivals) < 0
        ):
            raise ValueError("arrival_offsets must be nonnegative and monotonic")
        arrival_trace_source = "provided"
        arrival_trace_normalized = False

    worker_count = max(1, int(worker_count))
    microbatch_max_size = max(1, int(microbatch_max_size))
    try:
        direct_vllm_dispatch = bool(
            pipeline.config["direct_vllm_dispatch"]
        )
    except Exception:
        direct_vllm_dispatch = False
    reset_retrieval = getattr(pipeline, "reset_retrieval_batcher_stats", None)
    if callable(reset_retrieval):
        reset_retrieval()

    try:
        cross_requested = bool(
            pipeline.config["async_cross_microbatch_pipeline"]
        )
    except Exception:
        cross_requested = False
    try:
        retrieval_batch_size = max(
            1, int(pipeline.config["retrieval_batch_size"])
        )
    except Exception:
        retrieval_batch_size = microbatch_max_size
    cross_enabled = bool(
        cross_requested
        and not direct_vllm_dispatch
        and worker_count >= 2
        and microbatch_max_size <= retrieval_batch_size
    )
    if cross_enabled:
        reset_cross = getattr(
            pipeline, "reset_cross_microbatch_scheduler", None
        )
        if callable(reset_cross):
            reset_cross()

    admission_stats_getter = getattr(
        getattr(pipeline, "generator", None),
        "get_async_admission_stats",
        None,
    )
    request_queue = asyncio.Queue()
    stats = {
        "workflow": workflow_name,
        "batches": 0,
        "requests": 0,
        "request_queue_high_water": 0,
        "active_worker_high_water": 0,
        "microbatch_max_size": microbatch_max_size,
        "microbatch_policy": "worker_dispatch_drain",
        "request_batching_policy": "worker_nonblocking_drain",
        "preformed_work_queue": False,
        "microbatch_sizes": [],
        "arrival_trace_source": arrival_trace_source,
        "arrival_distribution": (
            "immediate"
            if np.isinf(arrival_rate)
            else "constant"
            if np.isinf(arrival_burstiness)
            else "gamma"
        ),
        "arrival_burstiness": float(arrival_burstiness),
        "arrival_trace_normalized": arrival_trace_normalized,
        "configured_arrival_rate_rps": float(arrival_rate),
        "requested_worker_count": worker_count,
        "effective_worker_count": worker_count,
        "workflow_concurrency_policy": "explicit_worker_count",
        "direct_vllm_dispatch": direct_vllm_dispatch,
        "application_generation_gate": not direct_vllm_dispatch,
        "retrieval_coalescing_policy": "ready_only_no_timed_window",
        "retrieval_intentional_wait_ms": 0.0,
        "generation_admission_policy": (
            "generator_max_num_seqs_and_prefill_token_budget"
        ),
        "cross_microbatch_requested": cross_requested,
        "cross_microbatch_pipeline": cross_enabled,
        "cross_microbatch_disabled_reason": (
            None
            if cross_enabled or not cross_requested
            else "superseded_by_direct_vllm_dispatch"
            if direct_vllm_dispatch
            else "worker_count_or_batch_shape"
        ),
        "exact_request_completion_timestamps": 0,
        "fallback_batch_completion_timestamps": 0,
    }
    batch_stage_runs = []
    active_workers = 0

    async def arrival_producer():
        for request_idx, (row, target) in enumerate(
            zip(source_rows, scheduled_arrivals)
        ):
            delay = float(target) - (time.perf_counter() - run_origin)
            if delay > 0:
                await asyncio.sleep(delay)
            await request_queue.put((request_idx, row))
            actual_arrival_offsets[request_idx] = time.perf_counter() - run_origin
            stats["request_queue_high_water"] = max(
                stats["request_queue_high_water"], request_queue.qsize()
            )
        for _ in range(worker_count):
            await request_queue.put(None)

    async def workflow_worker():
        nonlocal active_workers
        runner = workflow_runners[workflow_name]
        while True:
            first = await request_queue.get()
            if first is None:
                return
            batch_items = [first]
            stop_after_batch = False
            while len(batch_items) < microbatch_max_size:
                try:
                    item = request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                batch_items.append(item)

            indices, rows = zip(*batch_items)
            indices, rows = list(indices), list(rows)
            stats["microbatch_sizes"].append(len(rows))
            active_workers += 1
            stats["active_worker_high_water"] = max(
                stats["active_worker_high_water"], active_workers
            )
            try:
                output_dataset = await runner(Dataset(data=rows), do_eval=False)
                state = vars(output_dataset)
                stage_stats = state.get("_async_scheduler_stats")
                stage_trace = state.get("_async_scheduler_trace")
                if stage_stats is not None and stage_trace is not None:
                    batch_stage_runs.append(
                        {
                            "workflow_batch_id": len(batch_stage_runs),
                            "stats": dict(stage_stats),
                            "trace": [dict(record) for record in stage_trace],
                        }
                    )
                batch_finished = time.perf_counter() - run_origin
                for request_idx, output_item in zip(indices, output_dataset):
                    output = dict(output_item.output)
                    output_buffer[request_idx] = output
                    completed = output.get("generation_completion_monotonic_s")
                    if completed is None:
                        finish_offsets[request_idx] = batch_finished
                        stats["fallback_batch_completion_timestamps"] += 1
                    else:
                        finish_offsets[request_idx] = float(completed) - run_origin
                        stats["exact_request_completion_timestamps"] += 1
                stats["batches"] += 1
                stats["requests"] += len(rows)
            finally:
                active_workers -= 1
            if stop_after_batch:
                return

    workers = [asyncio.create_task(workflow_worker()) for _ in range(worker_count)]
    producer = asyncio.create_task(arrival_producer())
    try:
        await asyncio.gather(producer, *workers)
    except BaseException:
        for task in (producer, *workers):
            if not task.done():
                task.cancel()
        await asyncio.gather(producer, *workers, return_exceptions=True)
        raise

    if any(output is None for output in output_buffer):
        raise RuntimeError(
            f"Persistent {workflow_name} executor left unfinished requests"
        )
    if any(value is None for value in finish_offsets):
        raise RuntimeError(
            f"Persistent {workflow_name} executor left unfinished timings"
        )

    elapsed = time.perf_counter() - run_origin
    actual_arrivals = np.asarray(actual_arrival_offsets, dtype=float)
    finish_values = np.asarray(finish_offsets, dtype=float)
    arrival_schedule_lag = actual_arrivals - scheduled_arrivals
    post_admission = finish_values - actual_arrivals
    latency_values = finish_values - scheduled_arrivals
    stats.update(
        {
            "run_pipeline_time_s": elapsed,
            "throughput_qps": total_n / elapsed if elapsed else 0.0,
            "realized_arrival_rate_rps": (
                total_n / actual_arrivals[-1]
                if actual_arrivals[-1] > 0
                else 0.0
            ),
            "arrival_schedule_lag_mean_s": float(
                np.mean(arrival_schedule_lag)
            ),
            "arrival_schedule_lag_max_s": float(np.max(arrival_schedule_lag)),
            "e2e_latency_origin": "scheduled_arrival",
            "avg_e2e_s": float(np.mean(latency_values)),
            "p50_e2e_s": float(np.percentile(latency_values, 50)),
            "p95_e2e_s": float(np.percentile(latency_values, 95)),
            "p99_e2e_s": float(np.percentile(latency_values, 99)),
            "max_e2e_s": float(np.max(latency_values)),
            "post_admission_latency_mean_s": float(np.mean(post_admission)),
            "post_admission_latency_p95_s": float(
                np.percentile(post_admission, 95)
            ),
            "unique_finish_offsets": len(set(finish_offsets)),
            "completion_timestamp_mode": (
                "per_request_generation"
                if stats["exact_request_completion_timestamps"] == total_n
                else "mixed_or_batch_fallback"
            ),
        }
    )
    if callable(admission_stats_getter):
        stats["generation_admission"] = admission_stats_getter()
    retrieval_stats_getter = getattr(
        pipeline, "get_retrieval_batcher_stats", None
    )
    if callable(retrieval_stats_getter):
        stats["retrieval_batcher"] = retrieval_stats_getter()
    if batch_stage_runs:
        queue_capacities = {
            int(run["stats"]["generation_queue_capacity_batches"])
            for run in batch_stage_runs
            if run["stats"].get("generation_queue_capacity_batches") is not None
        }
        if len(queue_capacities) > 1:
            raise RuntimeError(
                "generation microbatch queue capacity changed within one workload"
            )
        if queue_capacities:
            stats["generation_queue_capacity_batches"] = queue_capacities.pop()
            stats["generation_queue_high_water_batches"] = max(
                int(run["stats"].get("generation_queue_high_water_batches") or 0)
                for run in batch_stage_runs
            )
    pipeline.last_workflow_scheduler_stats = stats

    cross_trace = getattr(pipeline, "_cross_microbatch_trace", None)
    if cross_enabled and cross_trace:
        summarize = getattr(pipeline, "summarize_scheduler_trace", None)
        if not callable(summarize):
            raise RuntimeError(
                "cross-microbatch scheduling requires trace summarization"
            )
        policy = str(pipeline.config["async_retrieval_policy"])
        cross_stats = summarize(
            cross_trace,
            policy,
            getattr(pipeline, "_cross_microbatch_adaptive_state", None),
        )
        cross_stats["workflow_batches"] = len(cross_trace)
        cross_stats["cross_microbatch_pipeline"] = True
        cross_stats["generation_queue_capacity_batches"] = stats.get(
            "generation_queue_capacity_batches"
        )
        cross_stats["generation_queue_high_water_batches"] = stats.get(
            "generation_queue_high_water_batches"
        )
        pipeline.last_async_scheduler_trace = list(cross_trace)
        pipeline.last_async_scheduler_stats = cross_stats
    elif batch_stage_runs:
        transition_pairs = sum(
            run["stats"].get("transition_pairs", 0)
            for run in batch_stage_runs
        )

        def weighted_mean(name):
            if not transition_pairs:
                return 0.0
            return sum(
                run["stats"].get(name, 0.0)
                * run["stats"].get("transition_pairs", 0)
                for run in batch_stage_runs
            ) / transition_pairs

        flattened = []
        for run in batch_stage_runs:
            for record in run["trace"]:
                record["workflow_batch_id"] = run["workflow_batch_id"]
                flattened.append(record)
        latest = batch_stage_runs[-1]["stats"]
        pipeline.last_async_scheduler_trace = flattened
        pipeline.last_async_scheduler_stats = {
            "policy": latest.get("policy"),
            "workflow_batches": len(batch_stage_runs),
            "batches": sum(
                run["stats"].get("batches", 0) for run in batch_stage_runs
            ),
            "transition_pairs": transition_pairs,
            "decode_overlap_pairs": sum(
                run["stats"].get("decode_overlap_pairs", 0)
                for run in batch_stage_runs
            ),
            "decode_contained_pairs": sum(
                run["stats"].get("decode_contained_pairs", 0)
                for run in batch_stage_runs
            ),
            "generation_contained_pairs": sum(
                run["stats"].get("generation_contained_pairs", 0)
                for run in batch_stage_runs
            ),
            "mean_retrieval_decode_coverage": weighted_mean(
                "mean_retrieval_decode_coverage"
            ),
            "mean_retrieval_prefill_coverage": weighted_mean(
                "mean_retrieval_prefill_coverage"
            ),
            "adaptive_estimates": latest.get("adaptive_estimates"),
            "configured_async_retrieval_policy": latest.get(
                "configured_async_retrieval_policy"
            ),
            "effective_async_retrieval_policy": latest.get(
                "effective_async_retrieval_policy"
            ),
            "direct_vllm_dispatch": direct_vllm_dispatch,
            "application_generation_gate": not direct_vllm_dispatch,
            "generation_queue_capacity_batches": stats.get(
                "generation_queue_capacity_batches"
            ),
            "generation_queue_high_water_batches": stats.get(
                "generation_queue_high_water_batches"
            ),
            "cross_microbatch_pipeline": False,
        }

    if record_request_timing:
        for request_idx, output in enumerate(output_buffer):
            dispatch = output.get("generation_dispatch_monotonic_s")
            output.update(
                {
                    "scheduled_arrival_offset_s": float(
                        scheduled_arrivals[request_idx]
                    ),
                    "arrival_offset_s": float(scheduled_arrivals[request_idx]),
                    "actual_arrival_offset_s": float(
                        actual_arrivals[request_idx]
                    ),
                    "arrival_schedule_lag_s": float(
                        arrival_schedule_lag[request_idx]
                    ),
                    "finish_offset_s": float(finish_offsets[request_idx]),
                    "e2e_latency_s": float(latency_values[request_idx]),
                    "post_admission_latency_s": float(
                        post_admission[request_idx]
                    ),
                }
            )
            if dispatch is not None:
                dispatch_offset = float(dispatch) - run_origin
                output["scheduler_admission_wait_s"] = max(
                    0.0, dispatch_offset - float(actual_arrivals[request_idx])
                )
                output["scheduler_queue_wait_s"] = max(
                    0.0, dispatch_offset - float(scheduled_arrivals[request_idx])
                )

    print(
        f"Persistent {workflow_name}: "
        f"batches={stats['batches']}, requests={stats['requests']}, "
        f"active_workers={stats['active_worker_high_water']}"
    )
    dataset = Dataset(data=source_rows)
    output_keys = sorted({key for output in output_buffer for key in output})
    for key in output_keys:
        dataset.update_output(key, [output.get(key) for output in output_buffer])
    return pipeline.evaluate(dataset, do_eval=do_eval)
