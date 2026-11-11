import asyncio
import time
import unittest

from mnemorag.pipeline.pipeline import AsyncPipeline


class FakeDataset:
    def __init__(self, questions):
        self.question = questions
        self.output = {}

    def update_output(self, name, values):
        self.output[name] = values


class FakeRetriever:
    def __init__(self, delay=0.02):
        self.delay = delay
        self.calls = []

    def batch_search(self, queries, **_):
        call = {"queries": list(queries), "start": time.perf_counter()}
        self.calls.append(call)
        time.sleep(self.delay)
        call["end"] = time.perf_counter()
        results = [["cold", "hot"] for _ in queries]
        scores = [[0.2, 0.9] for _ in queries]
        indices = [[1, 2] for _ in queries]
        return results, scores, indices


class FakeDocumentPool:
    def update(self, _):
        return None

    def reorder_by_hotness(self, doc_indices, documents):
        return list(reversed(documents)), list(reversed(doc_indices))

    def get_count(self, _doc_id):
        return 2


class FakePromptTemplate:
    def get_string(self, question, retrieval_result, return_document_spans=False, **_):
        prompt = f"{question}:{','.join(retrieval_result)}"
        if not return_document_spans:
            return prompt
        return prompt, [
            {
                "doc_position": index,
                "doc_id": document,
                "char_start": 0,
                "char_end": len(prompt),
            }
            for index, document in enumerate(retrieval_result)
        ]


class FakeGenerator:
    use_persistent_async_engine = True

    def __init__(self, prefill_delay=0.03, decode_delay=0.08):
        self.prefill_delay = prefill_delay
        self.decode_delay = decode_delay
        self.decode_barrier = None
        self.calls = []

    async def generate_async(
        self,
        prompts,
        first_token_event=None,
        prefill_start_event=None,
        **generation_params,
    ):
        call = {
            "prompts": list(prompts),
            "generation_params": dict(generation_params),
            "start": time.perf_counter(),
        }
        self.calls.append(call)
        call["prefill_start"] = time.perf_counter()
        if prefill_start_event is not None:
            prefill_start_event.mnemorag_set_time_s = call["prefill_start"]
            prefill_start_event.set()
        await asyncio.sleep(self.prefill_delay)
        call["first_token"] = time.perf_counter()
        if first_token_event is not None:
            first_token_event.mnemorag_set_time_s = call["first_token"]
            first_token_event.set()
        if self.decode_barrier is not None:
            await self.decode_barrier(len(self.calls) - 1)
        await asyncio.sleep(self.decode_delay)
        call["end"] = time.perf_counter()
        return [f"answer:{prompt}" for prompt in prompts]


class HangingGenerator:
    use_persistent_async_engine = True

    async def generate_async(self, _prompts, **_):
        await asyncio.Event().wait()


class FailingGenerator:
    use_persistent_async_engine = True

    async def generate_async(self, _prompts, **_):
        await asyncio.sleep(0.001)
        raise RuntimeError("generation failed")


class ShortGenerator:
    use_persistent_async_engine = True

    async def generate_async(self, prompts, first_token_event=None, **_):
        first_token_event.set()
        return ["answer"] * max(0, len(prompts) - 1)


def make_pipeline(
    policy,
    retrieval_delay=0.02,
    prefill_delay=0.03,
    decode_delay=0.08,
):
    pipeline = object.__new__(AsyncPipeline)
    pipeline.config = {
        "retrieval_batch_size": 2,
        "async_fixed_delay_ms": 10.0,
        "async_generation_queue_batches": 2,
        "async_adaptive_ema_alpha": 1.0,
        "async_adaptive_max_prefill_overlap_fraction": 0.5,
        "async_adaptive_safety_margin_ms": 5.0,
        "direct_vllm_dispatch": False,
    }
    if policy is not None:
        pipeline.config["async_retrieval_policy"] = policy
    pipeline.retriever = FakeRetriever(retrieval_delay)
    pipeline.query_pool = object()
    pipeline.document_pool = FakeDocumentPool()
    pipeline.prompt_template = FakePromptTemplate()
    pipeline.generator = FakeGenerator(prefill_delay, decode_delay)
    pipeline.RETRIEVAL_LOCK = asyncio.Lock()
    pipeline.GENERATOR_LOCK = asyncio.Lock()
    pipeline.evaluate = lambda dataset, **_: dataset
    return pipeline


class AsyncSchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def test_ready_retrieval_calls_coalesce_without_timed_window(self):
        pipeline = make_pipeline("eager", retrieval_delay=0.002)
        pipeline.config["retrieval_batch_size"] = 4

        first, second, third = await asyncio.gather(
            pipeline._retrieve_async(["q0"]),
            pipeline._retrieve_async(["q1"]),
            pipeline._retrieve_async(["q2"]),
        )

        self.assertEqual(len(pipeline.retriever.calls), 1)
        self.assertEqual(pipeline.retriever.calls[0]["queries"], ["q0", "q1", "q2"])
        self.assertEqual(first[2], [[1, 2]])
        self.assertEqual(second[2], [[1, 2]])
        self.assertEqual(third[2], [[1, 2]])
        stats = pipeline.get_retrieval_batcher_stats()
        self.assertEqual(stats["policy"], "ready_only_no_timed_window")
        self.assertEqual(stats["intentional_wait_ms"], 0.0)
        self.assertEqual(stats["retriever_batches"], 1)
        self.assertEqual(stats["coalesced_calls"], 2)

    async def test_direct_vllm_dispatch_disables_application_stage_gate(self):
        pipeline = make_pipeline("prefill_start")
        pipeline.config.update(
            {
                "direct_vllm_dispatch": True,
                "async_cross_microbatch_pipeline": True,
            }
        )
        pipeline.reset_cross_microbatch_scheduler()

        await asyncio.gather(
            pipeline.run_stream(FakeDataset(["q0"]), do_eval=False),
            pipeline.run_stream(FakeDataset(["q1"]), do_eval=False),
        )

        self.assertEqual(len(pipeline.generator.calls), 2)
        stats = pipeline.last_async_scheduler_stats
        self.assertTrue(stats["direct_vllm_dispatch"])
        self.assertFalse(stats["application_generation_gate"])
        self.assertEqual(stats["effective_async_retrieval_policy"], "eager")

    async def test_default_policy_waits_for_full_batch_prefill_admission(self):
        pipeline = make_pipeline(None)
        dataset = FakeDataset(["q0", "q1"])

        await pipeline.run_stream(dataset, do_eval=False)

        self.assertEqual(
            pipeline.last_async_scheduler_stats["policy"], "prefill_batch"
        )
        self.assertTrue(
            pipeline.generator.calls[0]["generation_params"][
                "prefill_start_after_all"
            ]
        )

    async def test_generation_microbatch_queue_applies_backpressure(self):
        pipeline = make_pipeline(
            "eager",
            retrieval_delay=0.001,
            prefill_delay=0.02,
            decode_delay=0.04,
        )
        pipeline.config["retrieval_batch_size"] = 1
        pipeline.config["async_generation_queue_batches"] = 2

        await pipeline.run_stream(
            FakeDataset([f"q{index}" for index in range(6)]),
            do_eval=False,
        )

        stats = pipeline.last_async_scheduler_stats
        self.assertEqual(stats["generation_queue_capacity_batches"], 2)
        self.assertEqual(stats["generation_queue_high_water_batches"], 2)

    async def test_recomp_uses_distinct_compression_and_answer_budgets(self):
        pipeline = make_pipeline("stage_aware")
        pipeline.config.update(
            {
                "generation_params": {"max_tokens": 32},
                "recomp_compression_max_tokens": 128,
                "recomp_answer_max_tokens": 32,
            }
        )

        class RecompPrompt:
            def get_string(
                self,
                question,
                retrieval_result=None,
                formatted_reference=None,
                **_,
            ):
                reference = (
                    formatted_reference
                    if formatted_reference is not None
                    else retrieval_result
                )
                return f"{question}:{reference}"

        pipeline.prompt_template = RecompPrompt()
        dataset = FakeDataset(["q0", "q1"])

        result = await pipeline.run_stream_recomp(dataset, do_eval=False)

        self.assertEqual(
            [
                call["generation_params"]["max_tokens"]
                for call in pipeline.generator.calls
            ],
            [128, 32],
        )
        self.assertEqual(len(result.output["compressed_doc"]), 2)
        self.assertEqual(len(result.output["pred"]), 2)
        self.assertEqual(len(result.output["generation_completion_monotonic_s"]), 2)
        self.assertEqual(len(result.output["generation_dispatch_monotonic_s"]), 2)
        self.assertTrue(
            all(
                dispatched <= completed
                for dispatched, completed in zip(
                    result.output["generation_dispatch_monotonic_s"],
                    result.output["generation_completion_monotonic_s"],
                )
            )
        )

    async def test_recomp_generation_overlaps_following_retrieval_chunk(self):
        pipeline = make_pipeline(
            "eager",
            retrieval_delay=0.03,
            prefill_delay=0.01,
            decode_delay=0.06,
        )

        class RecompPrompt:
            def get_string(
                self,
                question,
                retrieval_result=None,
                formatted_reference=None,
                **_,
            ):
                reference = (
                    formatted_reference
                    if formatted_reference is not None
                    else retrieval_result
                )
                return f"{question}:{reference}"

        pipeline.prompt_template = RecompPrompt()
        dataset = FakeDataset(["q0", "q1", "q2", "q3"])

        await pipeline.run_stream_recomp(dataset, do_eval=False)

        first, second = pipeline.last_recomp_async_trace
        self.assertLess(
            second["retrieval_start_s"], first["compression_end_s"]
        )

    async def test_recomp_outer_batches_wait_for_full_prefill_admission(self):
        pipeline = make_pipeline(
            "prefill_batch",
            retrieval_delay=0.01,
            prefill_delay=0.03,
            decode_delay=0.04,
        )
        pipeline.config.update(
            {
                "retrieval_batch_size": 8,
                "async_cross_microbatch_pipeline": True,
            }
        )

        class RecompPrompt:
            def get_string(
                self,
                question,
                retrieval_result=None,
                formatted_reference=None,
                **_,
            ):
                reference = (
                    formatted_reference
                    if formatted_reference is not None
                    else retrieval_result
                )
                return f"{question}:{reference}"

        pipeline.prompt_template = RecompPrompt()
        pipeline.reset_cross_microbatch_scheduler()

        await asyncio.gather(
            pipeline.run_stream_recomp(FakeDataset(["q0"]), do_eval=False),
            pipeline.run_stream_recomp(FakeDataset(["q1"]), do_eval=False),
        )

        retrievals = sorted(pipeline.retriever.calls, key=lambda call: call["start"])
        generations = sorted(pipeline.generator.calls, key=lambda call: call["start"])
        self.assertEqual(len(retrievals), 2)
        self.assertEqual(len(pipeline._cross_microbatch_trace), 2)
        self.assertGreaterEqual(
            retrievals[1]["start"], generations[0]["prefill_start"]
        )
        self.assertLess(retrievals[1]["start"], generations[0]["end"])

    async def test_irg_outer_batches_wait_for_full_prefill_admission(self):
        pipeline = make_pipeline(
            "prefill_batch",
            retrieval_delay=0.01,
            prefill_delay=0.03,
            decode_delay=0.04,
        )
        pipeline.config.update(
            {
                "retrieval_batch_size": 8,
                "async_cross_microbatch_pipeline": True,
                "irg_max_iter": 1,
                "irg_min_iter": 1,
            }
        )
        pipeline.reset_cross_microbatch_scheduler()

        await asyncio.gather(
            pipeline.run_stream_irg(FakeDataset(["q0"]), do_eval=False),
            pipeline.run_stream_irg(FakeDataset(["q1"]), do_eval=False),
        )

        retrievals = sorted(pipeline.retriever.calls, key=lambda call: call["start"])
        generations = sorted(pipeline.generator.calls, key=lambda call: call["start"])
        self.assertEqual(len(retrievals), 2)
        self.assertEqual(len(pipeline._cross_microbatch_trace), 2)
        self.assertGreaterEqual(
            retrievals[1]["start"], generations[0]["prefill_start"]
        )
        self.assertLess(retrievals[1]["start"], generations[0]["end"])

    async def test_irg_records_final_per_request_completion(self):
        pipeline = make_pipeline("stage_aware")
        pipeline.config.update({"irg_max_iter": 1, "irg_min_iter": 1})
        dataset = FakeDataset(["q0", "q1"])

        result = await pipeline.run_stream_irg(dataset, do_eval=False)

        self.assertEqual(result.output["irg_iterations"], [1, 1])
        self.assertEqual(len(result.output["generation_completion_monotonic_s"]), 2)
        self.assertTrue(all(result.output["generation_completion_monotonic_s"]))
        self.assertEqual(len(result.output["generation_dispatch_monotonic_s"]), 2)
        self.assertTrue(
            all(
                dispatched <= completed
                for dispatched, completed in zip(
                    result.output["generation_dispatch_monotonic_s"],
                    result.output["generation_completion_monotonic_s"],
                )
            )
        )

    async def test_stage_aware_retrieval_runs_inside_decode_window(self):
        pipeline = make_pipeline("stage_aware")
        dataset = FakeDataset([f"q{index}" for index in range(6)])

        async def wait_for_following_retrieval(call_index):
            if call_index >= 2:
                return

            async def retrieval_finished():
                while (
                    len(pipeline.retriever.calls) <= call_index + 1
                    or "end" not in pipeline.retriever.calls[call_index + 1]
                ):
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(retrieval_finished(), timeout=1.0)

        # Keep decode open until the following retrieval completes. This makes
        # the policy test independent of unrelated GC or host scheduling pauses.
        pipeline.generator.decode_barrier = wait_for_following_retrieval

        result = await pipeline.run_stream(dataset, do_eval=False)
        trace = pipeline.last_async_scheduler_trace

        self.assertEqual(len(trace), 3)
        self.assertEqual(
            [record["decode_ready_s"] for record in trace],
            [call["first_token"] for call in pipeline.generator.calls],
        )
        self.assertGreater(
            pipeline.last_async_scheduler_stats["mean_retrieval_decode_coverage"],
            0.9,
        )
        self.assertGreaterEqual(
            pipeline.last_async_scheduler_stats["mean_post_retrieval_preparation_s"],
            0.0,
        )
        self.assertGreaterEqual(
            pipeline.last_async_scheduler_stats["mean_decode_observer_delay_s"],
            0.0,
        )
        for current, following in zip(trace, trace[1:]):
            self.assertGreaterEqual(
                following["retrieval_start_s"], current["decode_ready_s"]
            )
            self.assertLess(following["retrieval_start_s"], current["generation_end_s"])
        self.assertEqual(result.output["retrieval_score"][0], [0.9, 0.2])
        self.assertTrue(all(result.output["pred"]))

    async def test_prefill_start_covers_prefill_and_decode_window(self):
        pipeline = make_pipeline(
            "prefill_start",
            retrieval_delay=0.04,
            prefill_delay=0.05,
            decode_delay=0.03,
        )
        dataset = FakeDataset([f"q{index}" for index in range(6)])

        await pipeline.run_stream(dataset, do_eval=False)
        trace = pipeline.last_async_scheduler_trace

        for current, following in zip(trace, trace[1:]):
            self.assertGreaterEqual(
                following["retrieval_start_s"], current["generation_start_s"]
            )
            self.assertLess(following["retrieval_start_s"], current["decode_ready_s"])
        stats = pipeline.last_async_scheduler_stats
        self.assertEqual(stats["generation_contained_pairs"], 2)
        self.assertGreater(stats["mean_retrieval_prefill_coverage"], 0.9)
        self.assertGreater(stats["mean_retrieval_generation_coverage"], 0.99)
        self.assertLess(stats["mean_uncovered_retrieval_s"], 0.002)
        self.assertGreater(stats["mean_ttft_from_dispatch_s"], 0.0)
        self.assertGreaterEqual(stats["mean_prefill_observer_delay_s"], 0.0)

    async def test_sequential_policy_disables_decode_overlap(self):
        pipeline = make_pipeline("sequential")
        dataset = FakeDataset([f"q{index}" for index in range(4)])

        await pipeline.run_stream(dataset, do_eval=False)
        first, second = pipeline.last_async_scheduler_trace

        self.assertGreaterEqual(second["retrieval_start_s"], first["generation_end_s"])
        self.assertEqual(
            pipeline.last_async_scheduler_stats["mean_retrieval_decode_coverage"],
            0.0,
        )

    async def test_cache_aware_prefill_batching_separates_seed_and_ready_requests(self):
        pipeline = make_pipeline(
            "prefill_start",
            retrieval_delay=0.001,
            prefill_delay=0.001,
            decode_delay=0.001,
        )
        pipeline.config.update({"cache_aware_prefill_batching": True})
        pipeline.enable_document_kv_heat = True
        dataset = FakeDataset(["q0", "q1", "q2", "q3"])

        result = await pipeline.run_stream(dataset, do_eval=False)

        self.assertEqual(
            [len(call["prompts"]) for call in pipeline.generator.calls],
            [1, 1, 2],
        )
        stats = pipeline._cache_aware_prefill_batching_stats
        self.assertEqual(stats["admission_batches"], 2)
        self.assertEqual(stats["cold_requests"], 1)
        self.assertEqual(stats["ready_requests"], 3)
        self.assertEqual(stats["cold_batches"], 1)
        self.assertEqual(stats["ready_batches"], 2)
        self.assertFalse(stats["completion_barrier"])
        self.assertEqual(stats["submission_policy"], "cold_then_ready_nonblocking")
        self.assertLess(
            pipeline.generator.calls[1]["start"],
            pipeline.generator.calls[0]["end"],
        )
        self.assertEqual(
            result.output["pred"],
            [
                "answer:q0:hot,cold",
                "answer:q1:hot,cold",
                "answer:q2:hot,cold",
                "answer:q3:hot,cold",
            ],
        )

    async def test_cache_aware_state_transition_is_atomic_across_workers(self):
        pipeline = make_pipeline(
            "eager",
            retrieval_delay=0.005,
            prefill_delay=0.005,
            decode_delay=0.005,
        )
        pipeline.config.update(
            {
                "cache_aware_prefill_batching": True,
                "direct_vllm_dispatch": True,
            }
        )
        pipeline.enable_document_kv_heat = True
        first = FakeDataset(["q0"])
        second = FakeDataset(["q1"])

        await asyncio.gather(
            pipeline.run_stream(first, do_eval=False),
            pipeline.run_stream(second, do_eval=False),
        )

        kinds = {
            first._async_scheduler_trace[0]["cache_aware_batch_kind"],
            second._async_scheduler_trace[0]["cache_aware_batch_kind"],
        }
        self.assertEqual(kinds, {"cold", "ready"})
        stats = pipeline._cache_aware_prefill_batching_stats
        self.assertEqual(stats["admission_batches"], 2)
        self.assertEqual(stats["cold_requests"], 1)
        self.assertEqual(stats["ready_requests"], 1)

    async def test_stage_aware_cannot_contain_retrieval_longer_than_decode(self):
        pipeline = make_pipeline(
            "stage_aware",
            retrieval_delay=0.08,
            prefill_delay=0.02,
            decode_delay=0.03,
        )
        dataset = FakeDataset([f"q{index}" for index in range(6)])

        await pipeline.run_stream(dataset, do_eval=False)
        stats = pipeline.last_async_scheduler_stats

        self.assertEqual(stats["transition_pairs"], 2)
        self.assertEqual(stats["decode_contained_pairs"], 0)
        self.assertGreater(stats["mean_retrieval_decode_coverage"], 0.0)
        self.assertLess(stats["mean_retrieval_decode_coverage"], 1.0)

    async def test_eager_uses_prefill_but_can_finish_before_generation(self):
        pipeline = make_pipeline(
            "eager",
            retrieval_delay=0.05,
            prefill_delay=0.03,
            decode_delay=0.04,
        )
        dataset = FakeDataset([f"q{index}" for index in range(6)])

        await pipeline.run_stream(dataset, do_eval=False)
        stats = pipeline.last_async_scheduler_stats

        self.assertEqual(stats["decode_contained_pairs"], 0)
        self.assertEqual(stats["generation_contained_pairs"], 2)
        self.assertGreater(stats["mean_retrieval_prefill_coverage"], 0.0)

    async def test_adaptive_policy_uses_bounded_prefill_lead(self):
        pipeline = make_pipeline(
            "adaptive",
            retrieval_delay=0.10,
            prefill_delay=0.06,
            decode_delay=0.04,
        )
        dataset = FakeDataset([f"q{index}" for index in range(6)])

        await pipeline.run_stream(dataset, do_eval=False)
        first, second, third = pipeline.last_async_scheduler_trace

        self.assertEqual(first["adaptive_gate"], "stage_aware_warmup")
        self.assertEqual(second["adaptive_gate"], "predictive")
        self.assertGreater(third["retrieval_start_s"], second["generation_start_s"])
        self.assertLess(
            second["adaptive_retrieval_target_s"],
            second["decode_ready_s"],
        )
        self.assertLess(third["retrieval_start_s"], second["generation_end_s"])
        self.assertGreater(second["adaptive_prefill_lead_s"], 0.0)
        self.assertLessEqual(
            second["adaptive_prefill_lead_s"],
            second["adaptive_prefill_ema_s"] * 0.5 + 1e-6,
        )
        stats = pipeline.last_async_scheduler_stats
        self.assertGreater(stats["mean_retrieval_decode_coverage"], 0.0)
        self.assertEqual(stats["transition_pairs"], 2)

    async def test_stage_gate_timeout_cancels_generation_worker(self):
        pipeline = make_pipeline("stage_aware")
        pipeline.config["async_scheduler_gate_timeout_s"] = 0.02
        pipeline.generator = HangingGenerator()
        dataset = FakeDataset(["q0", "q1", "q2", "q3"])

        with self.assertRaisesRegex(TimeoutError, "first token"):
            await pipeline.run_stream(dataset, do_eval=False)

    async def test_eager_propagates_generation_failure_without_deadlock(self):
        pipeline = make_pipeline("eager", retrieval_delay=0.002)
        pipeline.config["async_generation_queue_batches"] = 1
        pipeline.config["async_scheduler_gate_timeout_s"] = 0.2
        pipeline.generator = FailingGenerator()
        dataset = FakeDataset([f"q{index}" for index in range(12)])

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await asyncio.wait_for(
                pipeline.run_stream(dataset, do_eval=False), timeout=0.5
            )

    async def test_generator_output_count_must_match_prompt_count(self):
        pipeline = make_pipeline("stage_aware")
        pipeline.generator = ShortGenerator()
        dataset = FakeDataset(["q0", "q1"])

        with self.assertRaisesRegex(RuntimeError, "1 outputs for 2 prompts"):
            await pipeline.run_stream(dataset, do_eval=False)

    async def test_cross_microbatch_submits_generation_before_previous_decode_ends(
        self,
    ):
        pipeline = make_pipeline(
            "stage_aware",
            retrieval_delay=0.02,
            prefill_delay=0.03,
            decode_delay=0.08,
        )
        pipeline.config["retrieval_batch_size"] = 8
        pipeline.config["async_cross_microbatch_pipeline"] = True
        pipeline.reset_cross_microbatch_scheduler()
        first = FakeDataset(["q0", "q1"])
        second = FakeDataset(["q2", "q3"])

        await asyncio.gather(
            pipeline.run_stream(first, do_eval=False),
            pipeline.run_stream(second, do_eval=False),
        )

        retrievals = sorted(pipeline.retriever.calls, key=lambda call: call["start"])
        generations = sorted(pipeline.generator.calls, key=lambda call: call["start"])
        self.assertEqual(len(retrievals), 2)
        self.assertEqual(len(generations), 2)
        self.assertGreaterEqual(retrievals[1]["start"], generations[0]["first_token"])
        self.assertLess(retrievals[1]["start"], generations[0]["end"])
        self.assertLess(generations[1]["start"], generations[0]["end"])
        cross_stats = pipeline.summarize_scheduler_trace(
            pipeline._cross_microbatch_trace, "stage_aware"
        )
        self.assertEqual(cross_stats["transition_pairs"], 1)
        self.assertEqual(cross_stats["decode_contained_pairs"], 1)

    async def test_adaptive_state_persists_across_workflow_microbatches(self):
        pipeline = make_pipeline(
            "adaptive",
            retrieval_delay=0.10,
            prefill_delay=0.06,
            decode_delay=0.04,
        )
        pipeline.config["retrieval_batch_size"] = 8
        pipeline.config["async_cross_microbatch_pipeline"] = True
        pipeline.reset_cross_microbatch_scheduler()
        batches = [FakeDataset([f"q{index}"]) for index in range(3)]

        await asyncio.gather(
            *(pipeline.run_stream(batch, do_eval=False) for batch in batches)
        )

        first, second, third = pipeline._cross_microbatch_trace
        self.assertEqual(first["adaptive_gate"], "stage_aware_warmup")
        self.assertEqual(second["adaptive_gate"], "predictive")
        self.assertGreater(third["retrieval_start_s"], second["generation_start_s"])
        self.assertLess(third["retrieval_start_s"], second["decode_ready_s"])
        self.assertTrue(
            all(
                value is not None
                for value in pipeline._cross_microbatch_adaptive_state.values()
            )
        )

        pipeline.reset_cross_microbatch_scheduler()
        self.assertEqual(pipeline._cross_microbatch_trace, [])
        self.assertTrue(
            all(
                value is None
                for value in pipeline._cross_microbatch_adaptive_state.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
