import asyncio
import os
import time
from collections import deque
from itertools import chain
from pathlib import Path

from mnemorag.evaluator import Evaluator
from mnemorag.pipeline.utils import DocumentPool, QueryPool
from mnemorag.prompt import PromptTemplate
from mnemorag.utils import get_generator, get_retriever


class BasicPipeline:
    """Shared lifecycle and evaluation support for serving pipelines."""

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """Run inference for a dataset."""
        raise NotImplementedError

    def close(self):
        """Release persistent retrieval and generation resources."""
        retriever_close = getattr(self.retriever, "close", None)
        generator = getattr(self, "generator", None)
        generator_shutdown = getattr(generator, "shutdown", None)
        try:
            if callable(retriever_close):
                retriever_close()
        finally:
            if callable(generator_shutdown):
                generator_shutdown()

    def _bind_prompt_tokenizer(self):
        tokenizer = getattr(getattr(self, "generator", None), "tokenizer", None)
        bind_tokenizer = getattr(self.prompt_template, "bind_tokenizer", None)
        if tokenizer is not None and callable(bind_tokenizer):
            bind_tokenizer(tokenizer)

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """Evaluate predictions and optionally persist the retrieval cache."""
        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class IRGState:
    def __init__(self, question: str):
        self.question = question
        self.iter = 0
        self.generations = []
        self.prompts = []
        self.retrieval_queries = []
        self.retrievals = []
        self.retrieval_scores = []
        self.retrieval_idxs = []
        self.retrieval_times = []
        self.generation_times = []
        self.stop_reason = None


def _parse_cpu_set(value):
    cpus = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


class AsyncPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        initialization_started = time.perf_counter()
        initialization_profile = {}
        super().__init__(config, prompt_template)
        retriever_first = os.environ.get("MNEMORAG_RETRIEVER_FIRST", "0") == "1"
        if retriever_first:
            retriever_cpus = _parse_cpu_set(
                os.environ.get("MNEMORAG_RETRIEVER_CPUSET", "")
            )
            generator_cpus = _parse_cpu_set(
                os.environ.get("MNEMORAG_GENERATOR_CPUSET", "")
            )
            if retriever_cpus:
                os.sched_setaffinity(0, retriever_cpus)
                print(f"[MnemoRAG] Retriever CPU affinity: {sorted(retriever_cpus)}")
            phase_started = time.perf_counter()
            self.retriever = get_retriever(config) if retriever is None else retriever
            initialization_profile["retriever_s"] = time.perf_counter() - phase_started

            if generator_cpus:
                os.sched_setaffinity(0, generator_cpus)
                print(f"[MnemoRAG] Generator CPU affinity: {sorted(generator_cpus)}")
            phase_started = time.perf_counter()
            self.generator = get_generator(config) if generator is None else generator
            initialization_profile["generator_s"] = time.perf_counter() - phase_started
            if retriever_cpus:
                os.sched_setaffinity(0, retriever_cpus)
        else:
            phase_started = time.perf_counter()
            self.generator = get_generator(config) if generator is None else generator
            initialization_profile["generator_s"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            self.retriever = get_retriever(config) if retriever is None else retriever
            initialization_profile["retriever_s"] = time.perf_counter() - phase_started

        self._bind_prompt_tokenizer()

        query_base_capacity = int(self._config_or("query_base_capacity", 80_000))
        query_base_dtype = self._config_or("query_base_dtype", "float16")
        query_base_policy = self._config_or("query_base_policy", "lru")
        query_base_retain_metadata = bool(
            self._config_or("query_base_retain_metadata", False)
        )
        query_base_exact_lookup = bool(self._config_or("query_base_exact_lookup", True))
        self.query_pool = QueryPool(
            device=self.device,
            dtype=query_base_dtype,
            max_size=query_base_capacity,
            replacement_policy=query_base_policy,
            retain_query_metadata=query_base_retain_metadata,
            enable_exact_lookup=query_base_exact_lookup,
        )
        query_base_snapshot_path = os.environ.get(
            "MNEMORAG_QUERY_BASE_SNAPSHOT_PATH", ""
        )
        self.query_base_snapshot = {
            "enabled": bool(query_base_snapshot_path),
            "loaded_entries": 0,
        }
        if query_base_snapshot_path:
            snapshot_started = time.perf_counter()
            snapshot_metadata = self.query_pool.load_snapshot(query_base_snapshot_path)
            self.query_base_snapshot.update(
                {
                    "path": str(Path(query_base_snapshot_path).resolve()),
                    "loaded_entries": self.query_pool.current_size(),
                    "metadata": snapshot_metadata,
                    "load_s": time.perf_counter() - snapshot_started,
                }
            )
            initialization_profile["query_base_snapshot_s"] = self.query_base_snapshot[
                "load_s"
            ]
        print(
            f"[MnemoRAG] Query Base capacity={query_base_capacity}, "
            f"dtype={query_base_dtype}, policy={query_base_policy}, "
            f"retain_metadata={query_base_retain_metadata}, "
            f"exact_lookup={query_base_exact_lookup}"
        )
        try:
            enable_hotness_sort = bool(config["enable_hotness_sort"])
        except Exception:
            enable_hotness_sort = True
        dense_hotness = bool(self._config_or("document_hotness_dense_counts", True))
        try:
            document_count = len(self.retriever.corpus) if dense_hotness else None
        except Exception:
            document_count = None
        self.document_pool = DocumentPool(
            enable_hotness=enable_hotness_sort,
            num_documents=document_count,
        )
        self.enable_document_kv_heat = bool(
            self._config_or("enable_document_kv_heat", True)
        ) and bool(getattr(self.generator, "supports_document_heat_metadata", False))
        print(f"[MnemoRAG] enable_hotness_sort={enable_hotness_sort}")
        print(f"[MnemoRAG] enable_document_kv_heat={self.enable_document_kv_heat}")

        self.RETRIEVAL_LOCK = asyncio.Lock()
        self.GENERATOR_LOCK = asyncio.Lock()
        self._retrieval_batch_loop = None
        self._retrieval_batch_queue = None
        self._retrieval_batch_dispatcher_task = None
        self.reset_retrieval_batcher_stats()
        self.reset_cross_microbatch_scheduler()
        initialization_profile["total_s"] = time.perf_counter() - initialization_started
        retriever_profile = getattr(self.retriever, "initialization_profile", None)
        if retriever_profile is not None:
            initialization_profile["retriever_detail"] = dict(retriever_profile)
        self.initialization_profile = initialization_profile
        print(
            "[MnemoRAG] Pipeline initialization profile: "
            f"generator={initialization_profile['generator_s']:.3f}s, "
            f"retriever={initialization_profile['retriever_s']:.3f}s, "
            f"total={initialization_profile['total_s']:.3f}s"
        )
        self.first = True

    def _config_or(self, key, default):
        try:
            value = self.config[key]
        except Exception:
            return default
        return default if value is None else value

    def reset_retrieval_batcher_stats(self):
        """Reset measurement-only counters for ready-request coalescing."""
        task = getattr(self, "_retrieval_batch_dispatcher_task", None)
        queue = getattr(self, "_retrieval_batch_queue", None)
        if task is not None and not task.done():
            raise RuntimeError("cannot reset an active retrieval batcher")
        if queue:
            raise RuntimeError("cannot reset a nonempty retrieval batcher")
        self._retrieval_batcher_stats = {
            "policy": "ready_only_no_timed_window",
            "intentional_wait_ms": 0.0,
            "submitted_calls": 0,
            "submitted_queries": 0,
            "retriever_batches": 0,
            "coalesced_calls": 0,
            "max_calls_per_batch": 0,
            "max_queries_per_batch": 0,
            "queue_high_water_calls": 0,
            "queue_high_water_queries": 0,
        }

    def get_retrieval_batcher_stats(self):
        return dict(getattr(self, "_retrieval_batcher_stats", {}))

    def _ensure_retrieval_batcher(self):
        loop = asyncio.get_running_loop()
        active_task = getattr(self, "_retrieval_batch_dispatcher_task", None)
        previous_loop = getattr(self, "_retrieval_batch_loop", None)
        if previous_loop is not loop:
            if active_task is not None and not active_task.done():
                raise RuntimeError(
                    "retrieval batcher is still active on another event loop"
                )
            self._retrieval_batch_loop = loop
            self._retrieval_batch_queue = deque()
            self._retrieval_batch_dispatcher_task = None
            if not hasattr(self, "_retrieval_batcher_stats"):
                self.reset_retrieval_batcher_stats()
        elif getattr(self, "_retrieval_batch_queue", None) is None:
            self._retrieval_batch_queue = deque()
        return loop, self._retrieval_batch_queue

    async def _retrieve_async(self, queries):
        """Retrieve without waiting for future arrivals.

        Concurrent workflow calls are combined only when they are already
        queued in the current event-loop turn. One dispatcher owns the
        stateful QueryPool transition, while each completed slice is released
        immediately to its caller for vLLM admission.
        """
        loop, queue = self._ensure_retrieval_batcher()
        query_list = list(queries)
        if not query_list:
            return [], [], []
        future = loop.create_future()
        queue.append(
            {
                "queries": query_list,
                "future": future,
                "enqueued_s": time.perf_counter(),
            }
        )
        stats = self._retrieval_batcher_stats
        stats["submitted_calls"] += 1
        stats["submitted_queries"] += len(query_list)
        stats["queue_high_water_calls"] = max(
            stats["queue_high_water_calls"], len(queue)
        )
        stats["queue_high_water_queries"] = max(
            stats["queue_high_water_queries"],
            sum(len(item["queries"]) for item in queue),
        )

        dispatcher = getattr(self, "_retrieval_batch_dispatcher_task", None)
        if dispatcher is None or dispatcher.done():
            dispatcher = loop.create_task(self._drain_ready_retrievals())
            self._retrieval_batch_dispatcher_task = dispatcher

            def consume_dispatcher_result(done_task):
                if self._retrieval_batch_dispatcher_task is done_task:
                    self._retrieval_batch_dispatcher_task = None
                if not done_task.cancelled():
                    done_task.exception()

            dispatcher.add_done_callback(consume_dispatcher_result)
        return await future

    async def _drain_ready_retrievals(self):
        queue = self._retrieval_batch_queue
        max_queries = max(1, int(self.config["retrieval_batch_size"]))
        # Yield one event-loop turn, not a timed window. This lets callers that
        # are already runnable enqueue before the ready set is drained.
        await asyncio.sleep(0)
        while queue:
            calls = []
            query_count = 0
            while queue:
                candidate = queue[0]
                if candidate["future"].cancelled():
                    queue.popleft()
                    continue
                candidate_count = len(candidate["queries"])
                if calls and query_count + candidate_count > max_queries:
                    break
                calls.append(queue.popleft())
                query_count += candidate_count
                if query_count >= max_queries:
                    break
            if not calls:
                continue

            queries = [query for call in calls for query in call["queries"]]
            stats = self._retrieval_batcher_stats
            stats["retriever_batches"] += 1
            stats["coalesced_calls"] += max(0, len(calls) - 1)
            stats["max_calls_per_batch"] = max(stats["max_calls_per_batch"], len(calls))
            stats["max_queries_per_batch"] = max(
                stats["max_queries_per_batch"], len(queries)
            )
            try:
                raw_results, raw_scores, raw_idxs = await asyncio.to_thread(
                    self.retriever.batch_search,
                    queries,
                    query_pool=self.query_pool,
                    return_score=True,
                    return_idx=True,
                )
                if not (
                    len(raw_results) == len(raw_scores) == len(raw_idxs) == len(queries)
                ):
                    raise RuntimeError(
                        "retriever returned a different number of rows than submitted"
                    )
                offset = 0
                for call in calls:
                    count = len(call["queries"])
                    result = (
                        raw_results[offset : offset + count],
                        raw_scores[offset : offset + count],
                        raw_idxs[offset : offset + count],
                    )
                    offset += count
                    if not call["future"].done():
                        call["future"].set_result(result)
            except BaseException as error:
                for call in calls:
                    if not call["future"].done():
                        call["future"].set_exception(error)
                while queue:
                    call = queue.popleft()
                    if not call["future"].done():
                        call["future"].set_exception(error)
                raise

    def reset_cross_microbatch_scheduler(self):
        """Reset persistent stage state before starting a new workload."""
        self._cross_microbatch_previous_gate = None
        self._cross_microbatch_trace = []
        self._cross_microbatch_adaptive_state = {
            "retrieval_ema_s": None,
            "prefill_ema_s": None,
            "decode_ema_s": None,
        }
        self._cross_microbatch_observed_retrieval_batches = set()
        self._cross_microbatch_observed_generation_batches = set()
        # Workloads may be launched by separate asyncio.run() calls. Recreate
        # the lock so it is never retained from an earlier event loop or a
        # failed workload.
        self._cross_microbatch_lock = asyncio.Lock()

    @staticmethod
    def summarize_scheduler_trace(trace, policy, adaptive_state=None):
        overlap_coverages = []
        prefill_overlap_coverages = []
        generation_overlap_coverages = []
        uncovered_retrieval_s = []
        retrieval_durations_s = []
        generation_admission_delays_s = []
        ttft_from_dispatch_s = []
        prefill_durations_s = []
        decode_durations_s = []
        post_retrieval_preparation_s = []
        decode_observer_delay_s = []
        prefill_observer_delay_s = []
        decode_contained_pairs = 0
        generation_contained_pairs = 0
        for record in trace:
            retrieval_end = record.get("retrieval_end_s")
            prompt_ready = record.get("prompt_ready_s")
            if retrieval_end is not None and prompt_ready is not None:
                post_retrieval_preparation_s.append(prompt_ready - retrieval_end)
            observer_delay = record.get("decode_observer_delay_s")
            if observer_delay is not None:
                decode_observer_delay_s.append(observer_delay)
            observer_delay = record.get("prefill_observer_delay_s")
            if observer_delay is not None:
                prefill_observer_delay_s.append(observer_delay)
            generation_start = record.get("generation_start_s")
            generation_dispatch = record.get("generation_dispatch_s")
            decode_start = record.get("decode_ready_s")
            generation_end = record.get("generation_end_s")
            if generation_dispatch is not None and generation_start is not None:
                generation_admission_delays_s.append(
                    max(0.0, generation_start - generation_dispatch)
                )
            if generation_dispatch is not None and decode_start is not None:
                ttft_from_dispatch_s.append(
                    max(0.0, decode_start - generation_dispatch)
                )
            if generation_start is not None and decode_start is not None:
                prefill_durations_s.append(max(0.0, decode_start - generation_start))
            if decode_start is not None and generation_end is not None:
                decode_durations_s.append(max(0.0, generation_end - decode_start))
        for current, following in zip(trace, trace[1:]):
            generation_start = current.get("generation_start_s")
            decode_start = current.get("decode_ready_s")
            generation_end = current.get("generation_end_s")
            retrieval_start = following.get("retrieval_start_s")
            retrieval_end = following.get("retrieval_end_s")
            if (
                decode_start is None
                or generation_end is None
                or retrieval_start is None
                or retrieval_end is None
            ):
                continue
            overlap = max(
                0.0,
                min(generation_end, retrieval_end) - max(decode_start, retrieval_start),
            )
            retrieval_duration = max(1e-12, retrieval_end - retrieval_start)
            retrieval_durations_s.append(retrieval_duration)
            overlap_coverages.append(overlap / retrieval_duration)
            if generation_start is not None:
                prefill_overlap = max(
                    0.0,
                    min(decode_start, retrieval_end)
                    - max(generation_start, retrieval_start),
                )
                prefill_overlap_coverages.append(prefill_overlap / retrieval_duration)
                generation_overlap = max(
                    0.0,
                    min(generation_end, retrieval_end)
                    - max(generation_start, retrieval_start),
                )
                generation_overlap_coverages.append(
                    generation_overlap / retrieval_duration
                )
                uncovered_retrieval_s.append(
                    max(0.0, retrieval_duration - generation_overlap)
                )
            if retrieval_start >= decode_start and retrieval_end <= generation_end:
                decode_contained_pairs += 1
            if retrieval_end <= generation_end:
                generation_contained_pairs += 1

        pair_count = max(0, len(trace) - 1)
        return {
            "policy": policy,
            "batches": len(trace),
            "transition_pairs": pair_count,
            "decode_overlap_pairs": sum(
                coverage > 0.0 for coverage in overlap_coverages
            ),
            "decode_contained_pairs": decode_contained_pairs,
            "generation_contained_pairs": generation_contained_pairs,
            "mean_retrieval_decode_coverage": (
                sum(overlap_coverages) / len(overlap_coverages)
                if overlap_coverages
                else 0.0
            ),
            "mean_retrieval_prefill_coverage": (
                sum(prefill_overlap_coverages) / len(prefill_overlap_coverages)
                if prefill_overlap_coverages
                else 0.0
            ),
            "mean_retrieval_generation_coverage": (
                sum(generation_overlap_coverages) / len(generation_overlap_coverages)
                if generation_overlap_coverages
                else 0.0
            ),
            "mean_uncovered_retrieval_s": (
                sum(uncovered_retrieval_s) / len(uncovered_retrieval_s)
                if uncovered_retrieval_s
                else 0.0
            ),
            "mean_retrieval_duration_s": (
                sum(retrieval_durations_s) / len(retrieval_durations_s)
                if retrieval_durations_s
                else 0.0
            ),
            "mean_generation_admission_delay_s": (
                sum(generation_admission_delays_s) / len(generation_admission_delays_s)
                if generation_admission_delays_s
                else 0.0
            ),
            "mean_ttft_from_dispatch_s": (
                sum(ttft_from_dispatch_s) / len(ttft_from_dispatch_s)
                if ttft_from_dispatch_s
                else 0.0
            ),
            "mean_prefill_duration_s": (
                sum(prefill_durations_s) / len(prefill_durations_s)
                if prefill_durations_s
                else 0.0
            ),
            "mean_decode_duration_s": (
                sum(decode_durations_s) / len(decode_durations_s)
                if decode_durations_s
                else 0.0
            ),
            "mean_post_retrieval_preparation_s": (
                sum(post_retrieval_preparation_s) / len(post_retrieval_preparation_s)
                if post_retrieval_preparation_s
                else 0.0
            ),
            "mean_decode_observer_delay_s": (
                sum(decode_observer_delay_s) / len(decode_observer_delay_s)
                if decode_observer_delay_s
                else 0.0
            ),
            "mean_prefill_observer_delay_s": (
                sum(prefill_observer_delay_s) / len(prefill_observer_delay_s)
                if prefill_observer_delay_s
                else 0.0
            ),
            "adaptive_estimates": adaptive_state,
        }

    async def _generate_async_safe(
        self,
        prompts,
        first_token_event=None,
        prefill_start_event=None,
        prefill_start_after_all=False,
        document_heat_spans=None,
        return_completion_times=False,
        **generation_params,
    ):
        if not prompts:
            for event in (prefill_start_event, first_token_event):
                if event is not None:
                    event.mnemorag_set_time_s = time.perf_counter()
                    event.set()
            return ([], []) if return_completion_times else []
        if getattr(self.generator, "use_persistent_async_engine", False):
            if (
                getattr(self, "enable_document_kv_heat", False)
                and document_heat_spans is not None
            ):
                generation_params["document_heat_spans"] = document_heat_spans
            result = await self.generator.generate_async(
                prompts,
                first_token_event=first_token_event,
                prefill_start_event=prefill_start_event,
                prefill_start_after_all=prefill_start_after_all,
                return_completion_times=return_completion_times,
                **generation_params,
            )
            if not return_completion_times:
                return result
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and len(result[1]) == len(prompts)
            ):
                return result
            # Compatibility for generator implementations that accept extra
            # keyword arguments but do not expose per-request completion.
            completed = time.perf_counter()
            return result, [completed] * len(prompts)
        if prefill_start_event is not None:
            prefill_start_event.mnemorag_set_time_s = time.perf_counter()
            prefill_start_event.set()
        async with self.GENERATOR_LOCK:
            if (
                getattr(self, "enable_document_kv_heat", False)
                and document_heat_spans is not None
            ):
                generation_params["document_heat_spans"] = document_heat_spans
            outputs = await asyncio.to_thread(
                self.generator.generate, prompts, **generation_params
            )
        if first_token_event is not None:
            first_token_event.mnemorag_set_time_s = time.perf_counter()
            first_token_event.set()
        if return_completion_times:
            completed = time.perf_counter()
            return outputs, [completed] * len(prompts)
        return outputs

    def _build_prompt_with_document_heat(self, question, docs, **params):
        if not getattr(self, "enable_document_kv_heat", False):
            return self.prompt_template.get_string(
                question=question, retrieval_result=docs, **params
            ), None

        retrieval_idxs = params.pop("retrieval_idxs", None)
        prompt, document_spans = self.prompt_template.get_string(
            question=question,
            retrieval_result=docs,
            return_document_spans=True,
            **params,
        )
        heat_spans = []
        for span in document_spans:
            doc_position = int(span.get("doc_position", -1))
            if retrieval_idxs is not None and 0 <= doc_position < len(retrieval_idxs):
                heat_key = retrieval_idxs[doc_position]
            else:
                heat_key = span["doc_id"]
            heat_spans.append(
                {
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "heat": max(1, self.document_pool.get_count(heat_key)),
                }
            )
        return prompt, self._record_document_heat_spans(heat_spans)

    def _record_document_heat_spans(self, heat_spans):
        """Record metadata coverage for every heat-aware generation request."""
        heat_values = [float(span["heat"]) for span in heat_spans]
        self._document_heat_metadata_requests = (
            getattr(self, "_document_heat_metadata_requests", 0) + 1
        )
        self._document_heat_metadata_spans = getattr(
            self, "_document_heat_metadata_spans", 0
        ) + len(heat_values)
        above_one = sum(heat > 1.0 for heat in heat_values)
        self._document_heat_metadata_above_one_spans = (
            getattr(self, "_document_heat_metadata_above_one_spans", 0) + above_one
        )
        if above_one:
            self._document_heat_metadata_above_one_requests = (
                getattr(
                    self,
                    "_document_heat_metadata_above_one_requests",
                    0,
                )
                + 1
            )
        self._document_heat_metadata_max = max(
            getattr(self, "_document_heat_metadata_max", 0.0),
            max(heat_values, default=0.0),
        )
        return heat_spans

    def _document_heat_for_keys(self, keys):
        return max(
            (max(1, self.document_pool.get_count(key)) for key in (keys or ())),
            default=1,
        )

    def get_document_heat_metadata_stats(self):
        return {
            "requests": getattr(self, "_document_heat_metadata_requests", 0),
            "spans": getattr(self, "_document_heat_metadata_spans", 0),
            "above_one_requests": getattr(
                self, "_document_heat_metadata_above_one_requests", 0
            ),
            "above_one_spans": getattr(
                self, "_document_heat_metadata_above_one_spans", 0
            ),
            "max_heat": getattr(self, "_document_heat_metadata_max", 0.0),
        }

    def _workflow_max_tokens(self, config_key, default=None):
        if default is None:
            generation_params = self._config_or("generation_params", {})
            default = generation_params.get("max_tokens", 32)
        return int(self._config_or(config_key, default))

    def _reorder_retrieval_batch(self, results, scores, idxs):
        """Update document heat and keep documents, ids, and scores aligned."""
        self.document_pool.update(list(chain.from_iterable(idxs)))
        reordered_results = []
        reordered_scores = []
        reordered_idxs = []

        for docs, doc_scores, doc_idxs in zip(results, scores, idxs):
            hot_docs, hot_idxs = self.document_pool.reorder_by_hotness(doc_idxs, docs)
            score_queues = {}
            for doc_idx, score in zip(doc_idxs, doc_scores):
                score_queues.setdefault(doc_idx, []).append(score)
            hot_scores = [score_queues[doc_idx].pop(0) for doc_idx in hot_idxs]

            reordered_results.append(hot_docs)
            reordered_scores.append(hot_scores)
            reordered_idxs.append(hot_idxs)

        return reordered_results, reordered_scores, reordered_idxs

    def _build_irg_prompt(
        self,
        question,
        docs,
        previous_gen,
        iter_id,
        retrieval_idxs=None,
        return_document_heat=False,
    ):
        if previous_gen:
            round_question = (
                f"{question}\n\nCurrent answer draft:\n{previous_gen}\n\n"
                "Revise the draft using the newly retrieved evidence."
            )
        else:
            round_question = question
        prompt_params = {
            "original_question": question,
            "previous_gen": previous_gen,
            "iter_id": iter_id,
        }
        if return_document_heat:
            return self._build_prompt_with_document_heat(
                question=round_question,
                docs=docs,
                retrieval_idxs=retrieval_idxs,
                **prompt_params,
            )
        return self.prompt_template.get_string(
            question=round_question,
            retrieval_result=docs,
            **prompt_params,
        )

    def _build_recomp_prompt(
        self,
        question,
        docs,
        retrieval_idxs=None,
        return_document_heat=False,
    ):
        document_parts = [
            f"Document {idx + 1}: "
            f"{doc.get('contents', str(doc)) if isinstance(doc, dict) else str(doc)}"
            for idx, doc in enumerate(docs)
        ]
        docs_text = "\n".join(document_parts)
        prompt = (
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
        if not return_document_heat:
            return prompt
        if not getattr(self, "enable_document_kv_heat", False):
            return prompt, None

        documents_start = prompt.index("Documents:\n") + len("Documents:\n")
        heat_spans = []
        cursor = documents_start
        for doc_position, part in enumerate(document_parts):
            if retrieval_idxs is not None and doc_position < len(retrieval_idxs):
                heat_key = retrieval_idxs[doc_position]
            elif isinstance(docs[doc_position], dict):
                heat_key = docs[doc_position].get("id")
            else:
                heat_key = doc_position
            heat_spans.append(
                {
                    "char_start": cursor,
                    "char_end": cursor + len(part),
                    "heat": max(1, self.document_pool.get_count(heat_key)),
                }
            )
            cursor += len(part) + 1
        return prompt, self._record_document_heat_spans(heat_spans)

    def _build_recomp_answer_prompt(
        self,
        question,
        compressed_doc,
        retrieval_idxs=None,
        return_document_heat=False,
    ):
        no_evidence = str(compressed_doc).strip() in {
            "",
            "<NO_EVIDENCE>",
            "NO_EVIDENCE",
        }
        evidence = "" if no_evidence else str(compressed_doc)
        prompt = self.prompt_template.get_string(
            question=question,
            formatted_reference=evidence,
        )
        if not return_document_heat:
            return prompt
        if not getattr(self, "enable_document_kv_heat", False):
            return prompt, None

        heat_spans = []
        if evidence:
            matched_evidence = evidence
            start = prompt.find(matched_evidence)
            if start < 0:
                matched_evidence = evidence.rstrip()
                start = prompt.find(matched_evidence)
            if start >= 0 and matched_evidence:
                heat_spans.append(
                    {
                        "char_start": start,
                        "char_end": start + len(matched_evidence),
                        "heat": self._document_heat_for_keys(retrieval_idxs),
                    }
                )
        return prompt, self._record_document_heat_spans(heat_spans)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        run_st_time = time.time()
        input_query = dataset.question

        retrieval_start = time.time()
        retrieval_results, retrieval_scores, retrieval_idxs = (
            self.retriever.batch_search(
                input_query,
                query_pool=self.query_pool,
                return_score=True,
                return_idx=True,
            )
        )
        retrieval_end = time.time()
        print(
            f"[MnemoRAG] Retrieval time for {len(input_query)} samples: {retrieval_end - retrieval_start:.4f} seconds"
        )
        print(f"[MnemoRAG] Current query pool size: {self.query_pool.current_size()}")

        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_score", retrieval_scores)
        dataset.update_output("retrieval_idxs", retrieval_idxs)

        zip_question_retrieval_result_time_st = time.time()
        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]
        zip_question_retrieval_result_time_ed = time.time()
        print(
            f"[MnemoRAG] zip_question_retrieval_result_time: {zip_question_retrieval_result_time_ed - zip_question_retrieval_result_time_st:.4f} seconds"
        )

        dataset.update_output("prompt", input_prompts)

        generate_time_st = time.time()
        pred_answer_list = self.generator.generate(input_prompts)
        generate_time_ed = time.time()
        print(
            f"[MnemoRAG] generate time: {generate_time_ed - generate_time_st:.4f} seconds"
        )

        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )
        run_ed_time = time.time()
        print(f"[MnemoRAG] run total time: {run_ed_time - run_st_time:.4f} seconds")
        return dataset

    async def run_stream(self, dataset, do_eval=True, pred_process_fun=None):
        """Overlap retrieval for batch i+1 with generation for batch i."""
        queries = dataset.question
        R = self.config["retrieval_batch_size"]
        total_n = len(queries)
        cache_aware_prefill_batching = bool(
            self._config_or("cache_aware_prefill_batching", False)
        )
        direct_vllm_dispatch = bool(self._config_or("direct_vllm_dispatch", False))
        cross_microbatch_enabled = (
            bool(self._config_or("async_cross_microbatch_pipeline", True))
            and (cache_aware_prefill_batching or total_n <= R)
            and not direct_vllm_dispatch
        )
        configured_policy = str(
            self._config_or("async_retrieval_policy", "prefill_batch")
        ).lower()
        policy = "eager" if direct_vllm_dispatch else configured_policy
        if policy not in {
            "stage_aware",
            "prefill_start",
            "prefill_batch",
            "adaptive",
            "sequential",
            "fixed_delay",
            "eager",
        }:
            raise ValueError(f"Unsupported async_retrieval_policy: {policy}")
        fixed_delay_s = float(self._config_or("async_fixed_delay_ms", 60.0)) / 1000.0
        adaptive_alpha = min(
            1.0,
            max(0.01, float(self._config_or("async_adaptive_ema_alpha", 0.25))),
        )
        adaptive_max_prefill_fraction = min(
            1.0,
            max(
                0.0,
                float(
                    self._config_or("async_adaptive_max_prefill_overlap_fraction", 0.5)
                ),
            ),
        )
        adaptive_margin_s = max(
            0.0,
            float(self._config_or("async_adaptive_safety_margin_ms", 5.0)) / 1000.0,
        )
        gate_timeout_s = max(
            0.01,
            float(self._config_or("async_scheduler_gate_timeout_s", 120.0)),
        )

        all_predictions = [None] * total_n
        all_retrieval_results = [None] * total_n
        all_retrieval_scores = [None] * total_n
        all_retrieval_idxs = [None] * total_n
        all_prompts = [None] * total_n
        all_generation_dispatch_times = [None] * total_n
        all_generation_completion_times = [None] * total_n
        scheduler_trace = []
        if cross_microbatch_enabled:
            if not hasattr(self, "_cross_microbatch_adaptive_state"):
                self.reset_cross_microbatch_scheduler()
            adaptive_state = self._cross_microbatch_adaptive_state
            observed_retrieval_batches = (
                self._cross_microbatch_observed_retrieval_batches
            )
            observed_generation_batches = (
                self._cross_microbatch_observed_generation_batches
            )
            adaptive_trace = self._cross_microbatch_trace
        else:
            adaptive_state = {
                "retrieval_ema_s": None,
                "prefill_ema_s": None,
                "decode_ema_s": None,
            }
            observed_retrieval_batches = set()
            observed_generation_batches = set()
            adaptive_trace = scheduler_trace

        generation_tasks = []
        pending_completions = set()

        async def run_generation_batch(item):
            (
                batch_indices,
                batch_prompts,
                batch_document_heat_spans,
                first_token_event,
                generation_started_event,
                completion,
                trace,
            ) = item

            async def observe_prefill_start():
                await generation_started_event.wait()
                signal_time = getattr(
                    generation_started_event,
                    "mnemorag_set_time_s",
                    time.perf_counter(),
                )
                trace["prefill_admission_s"] = signal_time
                trace["generation_start_s"] = signal_time
                trace["prefill_observer_delay_s"] = max(
                    0.0, time.perf_counter() - signal_time
                )

            async def observe_decode_ready():
                await first_token_event.wait()
                signal_time = getattr(
                    first_token_event,
                    "mnemorag_set_time_s",
                    time.perf_counter(),
                )
                trace["decode_ready_s"] = signal_time
                trace["decode_observer_delay_s"] = max(
                    0.0, time.perf_counter() - signal_time
                )

            prefill_observer = asyncio.create_task(observe_prefill_start())
            decode_observer = asyncio.create_task(observe_decode_ready())
            try:
                outputs, completion_times = await self._generate_async_safe(
                    batch_prompts,
                    document_heat_spans=batch_document_heat_spans,
                    first_token_event=first_token_event,
                    prefill_start_event=generation_started_event,
                    prefill_start_after_all=(policy == "prefill_batch"),
                    return_completion_times=True,
                )
                if len(outputs) != len(batch_indices):
                    raise RuntimeError(
                        "Generator returned "
                        f"{len(outputs)} outputs for {len(batch_indices)} prompts"
                    )
                if not first_token_event.is_set():
                    first_token_event.set()
                if not generation_started_event.is_set():
                    generation_started_event.mnemorag_set_time_s = trace[
                        "generation_dispatch_s"
                    ]
                    generation_started_event.set()
                await asyncio.gather(prefill_observer, decode_observer)
                for qi, output, completed in zip(
                    batch_indices, outputs, completion_times
                ):
                    all_predictions[qi] = output
                    all_generation_completion_times[qi] = float(completed)
                trace["generation_end_s"] = max(
                    completion_times, default=time.perf_counter()
                )
                completion.set_result(None)
            except BaseException as error:
                if not generation_started_event.is_set():
                    generation_started_event.mnemorag_set_time_s = time.perf_counter()
                    generation_started_event.set()
                first_token_event.set()
                if not completion.done():
                    if isinstance(error, asyncio.CancelledError):
                        completion.cancel()
                    else:
                        completion.set_exception(error)
                raise
            finally:
                for observer in (prefill_observer, decode_observer):
                    if not observer.done():
                        observer.cancel()
                await asyncio.gather(
                    prefill_observer,
                    decode_observer,
                    return_exceptions=True,
                )

        def observe_completion(completion):
            pending_completions.discard(completion)
            if not completion.cancelled():
                completion.exception()

        async def enqueue_generation(item):
            for task in generation_tasks:
                if task.done() and not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        raise error
            (
                batch_indices,
                _,
                _,
                _,
                _,
                _,
                trace,
            ) = item
            trace["generation_dispatch_s"] = time.perf_counter()
            for qi in batch_indices:
                all_generation_dispatch_times[qi] = trace["generation_dispatch_s"]
            generation_tasks.append(asyncio.create_task(run_generation_batch(item)))
            # Let the task enter the shared AsyncLLM admission limiter now.
            await asyncio.sleep(0)

        def update_ema(name, value):
            previous = adaptive_state[name]
            adaptive_state[name] = (
                value
                if previous is None
                else adaptive_alpha * value + (1.0 - adaptive_alpha) * previous
            )

        def update_adaptive_estimates():
            for trace in adaptive_trace:
                batch_id = trace.get("cross_batch_id", trace["batch_id"])
                if (
                    batch_id not in observed_retrieval_batches
                    and "retrieval_end_s" in trace
                ):
                    update_ema(
                        "retrieval_ema_s",
                        trace["retrieval_end_s"] - trace["retrieval_start_s"],
                    )
                    observed_retrieval_batches.add(batch_id)
                if (
                    batch_id not in observed_generation_batches
                    and "generation_end_s" in trace
                    and "decode_ready_s" in trace
                ):
                    update_ema(
                        "prefill_ema_s",
                        trace["decode_ready_s"] - trace["generation_start_s"],
                    )
                    update_ema(
                        "decode_ema_s",
                        trace["generation_end_s"] - trace["decode_ready_s"],
                    )
                    observed_generation_batches.add(batch_id)

        async def wait_until_event_or_completion(event, completion, label):
            event_ready = asyncio.create_task(event.wait())
            done, _ = await asyncio.wait(
                {event_ready, completion},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=gate_timeout_s,
            )
            if not done:
                event_ready.cancel()
                await asyncio.gather(event_ready, return_exceptions=True)
                raise TimeoutError(
                    f"Timed out waiting for {label} or generation completion"
                )
            if completion in done:
                completion.result()
            if not event_ready.done():
                event_ready.cancel()
                await asyncio.gather(event_ready, return_exceptions=True)

        async def wait_until_first_token_or_completion(first_token_event, completion):
            await wait_until_event_or_completion(
                first_token_event, completion, "the first token"
            )

        async def wait_for_retrieval_window(
            first_token_event,
            generation_started_event,
            completion,
            trace,
        ):
            if policy == "eager":
                return
            if policy == "fixed_delay":
                await asyncio.wait_for(
                    generation_started_event.wait(), timeout=gate_timeout_s
                )
                trace.setdefault(
                    "generation_start_s",
                    getattr(
                        generation_started_event,
                        "mnemorag_set_time_s",
                        time.perf_counter(),
                    ),
                )
                remaining = (
                    trace["generation_start_s"] + fixed_delay_s - time.perf_counter()
                )
                if remaining <= 0:
                    return
                timer = asyncio.create_task(asyncio.sleep(remaining))
                done, _ = await asyncio.wait(
                    {timer, completion},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if completion in done:
                    completion.result()
                if not timer.done():
                    timer.cancel()
                    await asyncio.gather(timer, return_exceptions=True)
                return
            if policy == "sequential":
                await asyncio.wait_for(
                    asyncio.shield(completion), timeout=gate_timeout_s
                )
                return
            if policy in {"prefill_start", "prefill_batch"}:
                await wait_until_event_or_completion(
                    generation_started_event,
                    completion,
                    (
                        "first prefill admission"
                        if policy == "prefill_start"
                        else "full-batch prefill admission"
                    ),
                )
                if generation_started_event.is_set():
                    trace.setdefault(
                        "generation_start_s",
                        getattr(
                            generation_started_event,
                            "mnemorag_set_time_s",
                            time.perf_counter(),
                        ),
                    )
                return
            if policy == "stage_aware":
                await wait_until_first_token_or_completion(
                    first_token_event, completion
                )
                return

            update_adaptive_estimates()
            retrieval_ema = adaptive_state["retrieval_ema_s"]
            prefill_ema = adaptive_state["prefill_ema_s"]
            decode_ema = adaptive_state["decode_ema_s"]
            if None in (retrieval_ema, prefill_ema, decode_ema):
                trace["adaptive_gate"] = "stage_aware_warmup"
                await wait_until_first_token_or_completion(
                    first_token_event, completion
                )
                return

            required_lead = max(0.0, retrieval_ema + adaptive_margin_s - decode_ema)
            prefill_lead = min(
                required_lead,
                prefill_ema * adaptive_max_prefill_fraction,
            )
            delay_after_generation_start = max(0.0, prefill_ema - prefill_lead)
            trace["adaptive_gate"] = "predictive"
            trace["adaptive_retrieval_ema_s"] = retrieval_ema
            trace["adaptive_prefill_ema_s"] = prefill_ema
            trace["adaptive_decode_ema_s"] = decode_ema
            trace["adaptive_prefill_lead_s"] = prefill_lead

            await asyncio.wait_for(
                generation_started_event.wait(), timeout=gate_timeout_s
            )
            trace.setdefault(
                "generation_start_s",
                getattr(
                    generation_started_event,
                    "mnemorag_set_time_s",
                    time.perf_counter(),
                ),
            )
            trace["adaptive_retrieval_target_s"] = (
                trace["generation_start_s"] + delay_after_generation_start
            )
            remaining = trace["adaptive_retrieval_target_s"] - time.perf_counter()
            if remaining <= 0:
                return
            timer = asyncio.create_task(asyncio.sleep(remaining))
            decode_ready = asyncio.create_task(first_token_event.wait())
            done, _ = await asyncio.wait(
                {timer, decode_ready, completion},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=gate_timeout_s,
            )
            if not done:
                for task in (timer, decode_ready):
                    task.cancel()
                await asyncio.gather(timer, decode_ready, return_exceptions=True)
                raise TimeoutError("Timed out waiting for the adaptive retrieval gate")
            if completion in done:
                completion.result()
            for task in (timer, decode_ready):
                if not task.done():
                    task.cancel()
            await asyncio.gather(timer, decode_ready, return_exceptions=True)

        cross_lock_held = False
        cross_previous_gate = None

        async def acquire_cross_slot(wait_for_retrieval=True):
            nonlocal cross_lock_held, cross_previous_gate
            if not cross_microbatch_enabled:
                return
            if not hasattr(self, "_cross_microbatch_lock"):
                self.reset_cross_microbatch_scheduler()
            await self._cross_microbatch_lock.acquire()
            cross_lock_held = True
            cross_previous_gate = self._cross_microbatch_previous_gate
            if wait_for_retrieval and cross_previous_gate is not None:
                await wait_for_retrieval_window(*cross_previous_gate)

        async def release_cross_slot_with_gate(gate):
            nonlocal cross_lock_held, cross_previous_gate
            if not cross_microbatch_enabled:
                return
            self._cross_microbatch_previous_gate = gate
            self._cross_microbatch_lock.release()
            cross_lock_held = False
            cross_previous_gate = None

        async def enqueue_cache_aware_batch():
            nonlocal cross_lock_held, cross_previous_gate
            records = []
            retrieval_cursor = 0
            while retrieval_cursor < total_n:
                batch_indices = list(
                    range(
                        retrieval_cursor,
                        min(retrieval_cursor + R, total_n),
                    )
                )
                batch_queries = [queries[i] for i in batch_indices]
                retrieval_trace = {
                    "retrieval_start_s": time.perf_counter(),
                }
                raw_results, raw_scores, raw_idxs = await self._retrieve_async(
                    batch_queries
                )
                retrieval_trace["retrieval_end_s"] = time.perf_counter()
                retrieval_results, retrieval_scores, retrieval_idxs = (
                    self._reorder_retrieval_batch(raw_results, raw_scores, raw_idxs)
                )

                for bi, q_idx in enumerate(batch_indices):
                    all_retrieval_results[q_idx] = retrieval_results[bi]
                    all_retrieval_scores[q_idx] = retrieval_scores[bi]
                    all_retrieval_idxs[q_idx] = retrieval_idxs[bi]
                    prompt, document_heat_spans = self._build_prompt_with_document_heat(
                        question=queries[q_idx],
                        docs=retrieval_results[bi],
                        retrieval_idxs=retrieval_idxs[bi],
                        original_retrieval_result=raw_results[bi],
                    )
                    all_prompts[q_idx] = prompt
                    records.append(
                        {
                            "request_index": q_idx,
                            "prompt": prompt,
                            "document_heat_spans": document_heat_spans,
                            "retrieval_idxs": tuple(retrieval_idxs[bi]),
                            "retrieval_start_s": retrieval_trace["retrieval_start_s"],
                            "retrieval_end_s": retrieval_trace["retrieval_end_s"],
                            "prompt_ready_s": time.perf_counter(),
                        }
                    )
                retrieval_cursor += R

            # Refresh heat after the currently admitted batch is retrieved.
            # This lets the first occurrence seed CPU KV before a repeated
            # occurrence from the same admitted batch is generated.
            for record in records:
                spans = record["document_heat_spans"] or ()
                for span, doc_id in zip(spans, record["retrieval_idxs"]):
                    span["heat"] = max(1, self.document_pool.get_count(doc_id))
                record["document_heat"] = max(
                    (float(span["heat"]) for span in spans),
                    default=1.0,
                )

            seen_documents = getattr(self, "_cache_aware_generated_documents", None)
            if seen_documents is None:
                seen_documents = set()
                self._cache_aware_generated_documents = seen_documents
            seeded_in_batch = set()
            cold_records = []
            ready_records = []
            for record in records:
                key = record["retrieval_idxs"]
                if key in seen_documents or key in seeded_in_batch:
                    ready_records.append(record)
                else:
                    cold_records.append(record)
                    if record["document_heat"] > 1.0:
                        seeded_in_batch.add(key)

            # Submission order is the only application-level cache hint. Cold
            # requests are submitted first and ready requests follow without a
            # completion barrier; vLLM continuous batching owns actual prefill.
            cold_records.sort(
                key=lambda record: (
                    -record["document_heat"],
                    record["request_index"],
                )
            )
            ready_records.sort(
                key=lambda record: (
                    record["retrieval_idxs"],
                    record["request_index"],
                )
            )

            batching_stats = getattr(
                self,
                "_cache_aware_prefill_batching_stats",
                {
                    "admission_batches": 0,
                    "cold_requests": 0,
                    "ready_requests": 0,
                    "cold_batches": 0,
                    "ready_batches": 0,
                    "completion_barrier": False,
                    "submission_policy": "cold_then_ready_nonblocking",
                },
            )
            batching_stats["admission_batches"] += 1
            batching_stats["cold_requests"] += len(cold_records)
            batching_stats["ready_requests"] += len(ready_records)

            for batch_kind, ordered_records in (
                ("cold", cold_records),
                ("ready", ready_records),
            ):
                for start in range(0, len(ordered_records), R):
                    generation_records = ordered_records[start : start + R]
                    if not generation_records:
                        continue
                    batching_stats[f"{batch_kind}_batches"] += 1
                    trace = {
                        "batch_id": len(scheduler_trace),
                        "batch_size": len(generation_records),
                        "cache_aware_batch_kind": batch_kind,
                        "retrieval_start_s": min(
                            record["retrieval_start_s"] for record in generation_records
                        ),
                        "retrieval_end_s": max(
                            record["retrieval_end_s"] for record in generation_records
                        ),
                        "prompt_ready_s": max(
                            record["prompt_ready_s"] for record in generation_records
                        ),
                    }
                    scheduler_trace.append(trace)
                    if cross_microbatch_enabled:
                        trace["cross_batch_id"] = len(self._cross_microbatch_trace)
                        self._cross_microbatch_trace.append(trace)
                    first_token_event = asyncio.Event()
                    generation_started_event = asyncio.Event()
                    completion = asyncio.get_running_loop().create_future()
                    pending_completions.add(completion)
                    completion.add_done_callback(observe_completion)
                    if cross_microbatch_enabled:
                        if not cross_lock_held:
                            await acquire_cross_slot(wait_for_retrieval=False)
                    await enqueue_generation(
                        (
                            [record["request_index"] for record in generation_records],
                            [record["prompt"] for record in generation_records],
                            [
                                record["document_heat_spans"]
                                for record in generation_records
                            ],
                            first_token_event,
                            generation_started_event,
                            completion,
                            trace,
                        )
                    )
                    gate = (
                        first_token_event,
                        generation_started_event,
                        completion,
                        trace,
                    )
                    if cross_microbatch_enabled:
                        await release_cross_slot_with_gate(gate)
                    if batch_kind == "cold":
                        seen_documents.update(
                            record["retrieval_idxs"]
                            for record in generation_records
                            if record["document_heat"] > 1.0
                        )
            self._cache_aware_prefill_batching_stats = batching_stats

        idx = 0
        previous_gate = None
        try:
            if cross_microbatch_enabled:
                await acquire_cross_slot(wait_for_retrieval=True)
            if cache_aware_prefill_batching:
                await enqueue_cache_aware_batch()
                idx = total_n

            while idx < total_n:
                if previous_gate is not None:
                    await wait_for_retrieval_window(*previous_gate)

                batch_indices = list(range(idx, min(idx + R, total_n)))
                batch_queries = [queries[i] for i in batch_indices]
                trace = {
                    "batch_id": len(scheduler_trace),
                    "batch_size": len(batch_indices),
                    "retrieval_start_s": time.perf_counter(),
                }
                scheduler_trace.append(trace)
                if cross_microbatch_enabled:
                    trace["cross_batch_id"] = len(self._cross_microbatch_trace)
                    self._cross_microbatch_trace.append(trace)

                raw_results, raw_scores, raw_idxs = await self._retrieve_async(
                    batch_queries
                )
                trace["retrieval_end_s"] = time.perf_counter()
                retrieval_results, retrieval_scores, retrieval_idxs = (
                    self._reorder_retrieval_batch(raw_results, raw_scores, raw_idxs)
                )

                batch_prompts = []
                batch_document_heat_spans = []
                for bi, q_idx in enumerate(batch_indices):
                    all_retrieval_results[q_idx] = retrieval_results[bi]
                    all_retrieval_scores[q_idx] = retrieval_scores[bi]
                    all_retrieval_idxs[q_idx] = retrieval_idxs[bi]
                    prompt, document_heat_spans = self._build_prompt_with_document_heat(
                        question=queries[q_idx],
                        docs=retrieval_results[bi],
                        retrieval_idxs=retrieval_idxs[bi],
                        original_retrieval_result=raw_results[bi],
                    )
                    all_prompts[q_idx] = prompt
                    batch_prompts.append(prompt)
                    batch_document_heat_spans.append(document_heat_spans)
                trace["prompt_ready_s"] = time.perf_counter()

                first_token_event = asyncio.Event()
                generation_started_event = asyncio.Event()
                completion = asyncio.get_running_loop().create_future()
                pending_completions.add(completion)
                completion.add_done_callback(observe_completion)
                await enqueue_generation(
                    (
                        batch_indices,
                        batch_prompts,
                        batch_document_heat_spans,
                        first_token_event,
                        generation_started_event,
                        completion,
                        trace,
                    )
                )
                previous_gate = (
                    first_token_event,
                    generation_started_event,
                    completion,
                    trace,
                )
                if cross_lock_held:
                    await release_cross_slot_with_gate(previous_gate)

                idx += R

            if generation_tasks:
                await asyncio.gather(*generation_tasks)
        except BaseException:
            if cross_lock_held:
                self._cross_microbatch_lock.release()
                cross_lock_held = False
            for task in generation_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*generation_tasks, return_exceptions=True)
            for completion in tuple(pending_completions):
                if not completion.done():
                    completion.cancel()
            raise

        update_adaptive_estimates()
        self.last_async_scheduler_trace = scheduler_trace
        self.last_async_scheduler_stats = self.summarize_scheduler_trace(
            scheduler_trace, policy, adaptive_state
        )
        self.last_async_scheduler_stats.update(
            {
                "configured_async_retrieval_policy": configured_policy,
                "effective_async_retrieval_policy": policy,
                "direct_vllm_dispatch": direct_vllm_dispatch,
                "application_generation_gate": not direct_vllm_dispatch,
            }
        )
        dataset._async_scheduler_trace = scheduler_trace
        dataset._async_scheduler_stats = dict(self.last_async_scheduler_stats)

        dataset.update_output("retrieval_result", all_retrieval_results)
        dataset.update_output("retrieval_score", all_retrieval_scores)
        dataset.update_output("retrieval_idxs", all_retrieval_idxs)
        dataset.update_output("prompt", all_prompts)
        dataset.update_output(
            "generation_dispatch_monotonic_s",
            all_generation_dispatch_times,
        )
        dataset.update_output(
            "generation_completion_monotonic_s",
            all_generation_completion_times,
        )
        dataset.update_output("pred", all_predictions)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )
        return dataset

    async def run_stream_irg(
        self,
        dataset,
        do_eval: bool = True,
        pred_process_fun=None,
    ):
        """Run iterative retrieval-generation with per-request trajectories."""
        queries = dataset.question
        total_n = len(queries)
        R = self.config["retrieval_batch_size"]
        max_iter = int(self._config_or("irg_max_iter", 2))
        min_iter = int(self._config_or("irg_min_iter", 1))
        early_stop = bool(self._config_or("irg_early_stop", False))
        if max_iter < 1:
            raise ValueError("irg_max_iter must be at least 1")
        if min_iter < 1 or min_iter > max_iter:
            raise ValueError("irg_min_iter must be in [1, irg_max_iter]")

        states = [IRGState(q) for q in queries]
        active = set(range(total_n))
        final_dispatch_times = [None] * total_n
        final_completion_times = [None] * total_n

        for irg_round in range(max_iter):
            if not active:
                break
            round_tasks = []
            for offset in range(0, len(active), R):
                batch_indices = sorted(active)[offset : offset + R]
                batch_queries = [
                    states[qi].question
                    if not states[qi].generations
                    else (
                        f"{states[qi].question}\n\n"
                        f"Current answer draft: {states[qi].generations[-1]}"
                    )
                    for qi in batch_indices
                ]
                retrieval_started = time.perf_counter()
                raw_results, raw_scores, raw_idxs = await self._retrieve_async(
                    batch_queries
                )
                retrieval_time = time.perf_counter() - retrieval_started
                results, scores, idxs = self._reorder_retrieval_batch(
                    raw_results, raw_scores, raw_idxs
                )

                prompts = []
                document_heat_spans = []
                for qi, retrieval_query, docs, doc_scores, doc_idxs in zip(
                    batch_indices, batch_queries, results, scores, idxs
                ):
                    state = states[qi]
                    state.retrieval_queries.append(str(retrieval_query))
                    state.retrievals.append(docs)
                    state.retrieval_scores.append(doc_scores)
                    state.retrieval_idxs.append(doc_idxs)
                    state.retrieval_times.append(retrieval_time)
                    prompt, heat_spans = self._build_irg_prompt(
                        state.question,
                        docs,
                        state.generations[-1] if state.generations else None,
                        state.iter + 1,
                        retrieval_idxs=doc_idxs,
                        return_document_heat=True,
                    )
                    state.prompts.append(prompt)
                    prompts.append(prompt)
                    document_heat_spans.append(heat_spans)

                async def run_generation(
                    q_indices,
                    batch_prompts,
                    batch_document_heat_spans,
                ):
                    generation_dispatched = time.perf_counter()
                    generation_started = time.perf_counter()
                    outputs, completion_times = await self._generate_async_safe(
                        batch_prompts,
                        document_heat_spans=batch_document_heat_spans,
                        max_tokens=self._workflow_max_tokens("irg_round_max_tokens"),
                        return_completion_times=True,
                    )
                    generation_time = time.perf_counter() - generation_started
                    return (
                        list(zip(q_indices, outputs, completion_times)),
                        generation_time,
                        generation_dispatched,
                    )

                round_tasks.append(
                    asyncio.create_task(
                        run_generation(
                            batch_indices,
                            prompts,
                            document_heat_spans,
                        )
                    )
                )

            if not round_tasks:
                raise RuntimeError(
                    f"IRG round {irg_round + 1} produced no generation work"
                )

            generated = await asyncio.gather(*round_tasks)
            next_active = set()
            for (
                batch_outputs,
                generation_time,
                generation_dispatched,
            ) in generated:
                for qi, output, completion_time in batch_outputs:
                    state = states[qi]
                    final_dispatch_times[qi] = float(generation_dispatched)
                    final_completion_times[qi] = float(completion_time)
                    previous_output = (
                        state.generations[-1] if state.generations else None
                    )
                    previous_idxs = (
                        state.retrieval_idxs[-2]
                        if len(state.retrieval_idxs) > 1
                        else None
                    )
                    state.generations.append(output)
                    state.generation_times.append(generation_time)
                    state.iter += 1

                    stable_output = (
                        previous_output is not None
                        and " ".join(str(previous_output).split()).lower()
                        == " ".join(str(output).split()).lower()
                    )
                    stable_retrieval = previous_idxs is not None and set(
                        previous_idxs
                    ) == set(state.retrieval_idxs[-1])
                    if (
                        early_stop
                        and state.iter >= min_iter
                        and stable_output
                        and stable_retrieval
                    ):
                        state.stop_reason = "stable_generation_and_retrieval"
                    elif state.iter >= max_iter:
                        state.stop_reason = "max_iter"
                    else:
                        next_active.add(qi)

            print(
                f"[MnemoRAG] IRG iter {irg_round + 1}/{max_iter} completed "
                f"for {len(active)} samples; continuing={len(next_active)}"
            )
            active = next_active

        if any(not state.generations for state in states):
            missing = sum(not state.generations for state in states)
            raise RuntimeError(f"IRG left {missing} requests unfinished")

        dataset.update_output("pred_trajectory", [s.generations for s in states])
        dataset.update_output("pred", [s.generations[-1] for s in states])
        dataset.update_output("prompt", [s.prompts[-1] for s in states])
        dataset.update_output("prompt_trajectory", [s.prompts for s in states])
        dataset.update_output("retrieval_result", [s.retrievals[-1] for s in states])
        dataset.update_output(
            "retrieval_score", [s.retrieval_scores[-1] for s in states]
        )
        dataset.update_output("retrieval_idxs", [s.retrieval_idxs[-1] for s in states])
        dataset.update_output(
            "retrieval_result_trajectory", [s.retrievals for s in states]
        )
        dataset.update_output(
            "retrieval_score_trajectory", [s.retrieval_scores for s in states]
        )
        dataset.update_output(
            "retrieval_idxs_trajectory", [s.retrieval_idxs for s in states]
        )
        dataset.update_output(
            "retrieval_query_trajectory", [s.retrieval_queries for s in states]
        )
        dataset.update_output("irg_iterations", [s.iter for s in states])
        dataset.update_output("irg_stop_reason", [s.stop_reason for s in states])
        dataset.update_output(
            "generation_dispatch_monotonic_s",
            final_dispatch_times,
        )
        dataset.update_output(
            "generation_completion_monotonic_s",
            final_completion_times,
        )
        dataset.update_output(
            "irg_retrieval_time_trajectory_s",
            [s.retrieval_times for s in states],
        )
        dataset.update_output(
            "irg_generation_time_trajectory_s",
            [s.generation_times for s in states],
        )
        return self.evaluate(
            dataset,
            do_eval=do_eval,
            pred_process_fun=pred_process_fun,
        )

    async def run_stream_recomp(self, dataset, do_eval=True, pred_process_fun=None):
        """Run RECOMP as retrieval, evidence compression, then answering."""
        questions = dataset.question
        retrieval_st = time.time()
        raw_results, raw_scores, raw_idxs = await self._retrieve_async(questions)
        retrieval_results, retrieval_scores, retrieval_idxs = (
            self._reorder_retrieval_batch(raw_results, raw_scores, raw_idxs)
        )
        retrieval_ed = time.time()
        print(f"[MnemoRAG] RECOMP retrieval time: {retrieval_ed - retrieval_st:.4f}s")

        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_score", retrieval_scores)
        dataset.update_output("retrieval_idxs", retrieval_idxs)
        dataset.update_output("original_retrieval_result", raw_results)
        dataset.update_output("original_retrieval_idxs", raw_idxs)

        compress_prompt_records = [
            self._build_recomp_prompt(
                question,
                docs,
                retrieval_idxs=doc_idxs,
                return_document_heat=True,
            )
            for question, docs, doc_idxs in zip(
                questions,
                retrieval_results,
                retrieval_idxs,
            )
        ]
        compress_prompts = [record[0] for record in compress_prompt_records]
        compress_document_heat_spans = [record[1] for record in compress_prompt_records]
        dataset.update_output("compression_prompt", compress_prompts)

        compress_st = time.time()
        compressed_docs = await self._generate_async_safe(
            compress_prompts,
            document_heat_spans=compress_document_heat_spans,
            max_tokens=self._workflow_max_tokens("recomp_compression_max_tokens", 128),
        )
        compress_ed = time.time()
        print(
            "[MnemoRAG] RECOMP compression generation time: "
            f"{compress_ed - compress_st:.4f}s"
        )

        dataset.update_output("compressed_doc", compressed_docs)
        dataset.update_output(
            "recomp_selected_augmentation",
            [
                str(compressed_doc).strip() not in {"", "<NO_EVIDENCE>", "NO_EVIDENCE"}
                for compressed_doc in compressed_docs
            ],
        )
        print("[MnemoRAG] RECOMP compressed evidence recorded", flush=True)

        print("[MnemoRAG] RECOMP building answer prompts", flush=True)
        answer_prompt_records = [
            self._build_recomp_answer_prompt(
                question,
                compressed_doc,
                retrieval_idxs=doc_idxs,
                return_document_heat=True,
            )
            for question, compressed_doc, doc_idxs in zip(
                questions,
                compressed_docs,
                retrieval_idxs,
            )
        ]
        answer_prompts = [record[0] for record in answer_prompt_records]
        answer_document_heat_spans = [record[1] for record in answer_prompt_records]
        dataset.update_output("prompt", answer_prompts)
        print("[MnemoRAG] RECOMP answer prompts ready", flush=True)

        answer_st = time.time()
        print("[MnemoRAG] RECOMP submitting answer generation", flush=True)
        answer_dispatch_monotonic_s = time.perf_counter()
        pred_answer_list, answer_completion_times = await self._generate_async_safe(
            answer_prompts,
            document_heat_spans=answer_document_heat_spans,
            max_tokens=self._workflow_max_tokens("recomp_answer_max_tokens"),
            return_completion_times=True,
        )
        answer_ed = time.time()
        print(f"[MnemoRAG] RECOMP answer generation time: {answer_ed - answer_st:.4f}s")

        dataset.update_output("pred", pred_answer_list)
        dataset.update_output(
            "generation_dispatch_monotonic_s",
            [answer_dispatch_monotonic_s for _ in questions],
        )
        dataset.update_output(
            "generation_completion_monotonic_s",
            answer_completion_times,
        )
        dataset.update_output(
            "recomp_retrieval_batch_time_s",
            [retrieval_ed - retrieval_st for _ in questions],
        )
        dataset.update_output(
            "recomp_compression_batch_time_s",
            [compress_ed - compress_st for _ in questions],
        )
        dataset.update_output(
            "recomp_answer_batch_time_s",
            [answer_ed - answer_st for _ in questions],
        )
        return self.evaluate(
            dataset,
            do_eval=do_eval,
            pred_process_fun=pred_process_fun,
        )

    async def run_stream_dc(self, dataset, do_eval=True, pred_process_fun=None):
        """Backward-compatible alias for the RECOMP workflow."""
        return await self.run_stream_recomp(
            dataset,
            do_eval=do_eval,
            pred_process_fun=pred_process_fun,
        )
