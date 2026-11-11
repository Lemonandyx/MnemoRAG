import asyncio
import time

from examples.quick_start.utils import poisson_persistent_workflow_run
from mnemorag.dataset.dataset import Dataset


class FakeGenerator:
    def get_async_admission_stats(self):
        return {"max_num_seqs": 8, "inflight_sequences": 0}


class FakePipeline:
    def __init__(self):
        self.config = {
            "direct_vllm_dispatch": True,
            "async_cross_microbatch_pipeline": False,
            "retrieval_batch_size": 2,
        }
        self.generator = FakeGenerator()
        self.last_workflow_scheduler_stats = None
        self.retrieval_resets = 0

    def reset_retrieval_batcher_stats(self):
        self.retrieval_resets += 1

    def get_retrieval_batcher_stats(self):
        return {
            "policy": "ready_only_no_timed_window",
            "intentional_wait_ms": 0.0,
        }

    async def run_stream(self, dataset, do_eval=False):
        await asyncio.sleep(0.001)
        now = time.perf_counter()
        dataset.update_output("pred", [f"answer:{q}" for q in dataset.question])
        dataset.update_output(
            "generation_dispatch_monotonic_s",
            [now - 0.0005 for _ in dataset.question],
        )
        dataset.update_output(
            "generation_completion_monotonic_s",
            [now for _ in dataset.question],
        )
        dataset._async_scheduler_stats = {
            "policy": "eager",
            "batches": 1,
            "transition_pairs": 0,
        }
        dataset._async_scheduler_trace = []
        return dataset

    run_stream_irg = run_stream
    run_stream_recomp = run_stream

    def evaluate(self, dataset, **_):
        return dataset


def test_online_scheduler_is_importable_and_preserves_request_order():
    async def run_test():
        pipeline = FakePipeline()
        dataset = Dataset(data=[{"question": f"q{i}"} for i in range(4)])

        result = await poisson_persistent_workflow_run(
            dataset,
            pipeline,
            float("inf"),
            workflow="ONEPASS",
            worker_count=2,
            microbatch_max_size=2,
            record_request_timing=True,
        )

        assert [item.output["pred"] for item in result] == [
            "answer:q0",
            "answer:q1",
            "answer:q2",
            "answer:q3",
        ]
        stats = pipeline.last_workflow_scheduler_stats
        assert stats["requests"] == 4
        assert stats["request_batching_policy"] == "worker_nonblocking_drain"
        assert stats["microbatch_policy"] == "worker_dispatch_drain"
        assert stats["preformed_work_queue"] is False
        assert stats["effective_worker_count"] == 2
        assert pipeline.retrieval_resets == 1

    asyncio.run(run_test())


def test_online_scheduler_routes_every_paper_workflow():
    async def run_test():
        for workflow in ("ONEPASS", "RECOMP", "IRG"):
            pipeline = FakePipeline()
            dataset = Dataset(data=[{"question": "q0"}, {"question": "q1"}])

            result = await poisson_persistent_workflow_run(
                dataset,
                pipeline,
                float("inf"),
                workflow=workflow,
                worker_count=2,
                microbatch_max_size=1,
            )

            assert len(result) == 2
            assert pipeline.last_workflow_scheduler_stats["workflow"] == workflow
            assert pipeline.last_workflow_scheduler_stats["requests"] == 2

    asyncio.run(run_test())
