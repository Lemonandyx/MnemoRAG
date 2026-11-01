import asyncio
import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnectorMetadata,
    OffloadingConnectorStats,
    OffloadingConnectorScheduler,
    OffloadingConnectorWorker,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids


def test_engine_core_exposes_block_pool_stats():
    expected = {"pattern_heat_entries": 17}
    block_pool = SimpleNamespace(get_mnemorag_stats=lambda: expected)
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=block_pool)
    )

    assert core.get_mnemorag_kv_cache_stats() == expected


def test_engine_core_merges_cpu_kv_transfer_stats():
    transfer = {"h2d_reload": {"completed_requests": 2}}
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=None),
        connector=SimpleNamespace(
            get_mnemorag_transfer_stats=lambda: transfer
        ),
    )

    assert core.get_mnemorag_kv_cache_stats() == {
        "cpu_kv_transfer": transfer
    }


def test_offload_scheduler_records_worker_transfer_and_h2d_wait_latency():
    scheduler = OffloadingConnectorScheduler.__new__(
        OffloadingConnectorScheduler
    )
    scheduler._h2d_wait_started_s = {}
    scheduler._worker_duration_samples = {
        False: deque(maxlen=4096),
        True: deque(maxlen=4096),
    }
    scheduler._transfer_stats = {
        False: {
            "scheduled_request_operations": 0,
            "scheduled_spec_units": 0,
            "worker_completed_jobs": 0,
            "worker_completed_requests": 0,
            "worker_transferred_blocks": 0,
            "worker_observed_total_s": 0.0,
            "worker_observed_max_s": 0.0,
            "scheduler_wait_completed_requests": 0,
            "scheduler_wait_total_s": 0.0,
            "scheduler_wait_max_s": 0.0,
        },
        True: {
            "scheduled_request_operations": 0,
            "scheduled_spec_units": 0,
            "worker_completed_jobs": 0,
            "worker_completed_requests": 0,
            "worker_transferred_blocks": 0,
            "worker_observed_total_s": 0.0,
            "worker_observed_max_s": 0.0,
        },
    }
    transfer = {
        "request-1": (
            SimpleNamespace(block_ids=[1, 2]),
            SimpleNamespace(block_ids=[3, 4]),
        )
    }

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "offloading_connector.time.perf_counter",
        side_effect=[10.0, 10.25, 20.0],
    ):
        scheduler._record_transfer_starts(transfer, store=False)
        scheduler._record_h2d_wait_finish("request-1")
        scheduler._record_transfer_starts(transfer, store=True)

    worker_stats = OffloadingConnectorStats()
    worker_stats.record_transfer(
        store=False, duration_s=0.10, request_count=1, block_count=2
    )
    worker_stats.record_transfer(
        store=True, duration_s=0.50, request_count=1, block_count=2
    )
    scheduler._consume_worker_transfer_stats(worker_stats)

    stats = scheduler.get_mnemorag_transfer_stats()
    assert stats["h2d_reload"]["worker_observed_mean_s"] == 0.10
    assert stats["d2h_offload"]["worker_observed_mean_s"] == 0.50
    assert stats["h2d_reload"]["scheduler_wait_mean_s"] == 0.25
    assert stats["h2d_reload"]["scheduled_spec_units"] == 2
    assert stats["d2h_offload"]["worker_transferred_blocks"] == 2


def test_offload_worker_batches_requests_and_records_completion_latency():
    class Spec:
        def __init__(self, block_ids):
            self.block_ids = list(block_ids)

    class TransferWorker:
        def __init__(self):
            self.submitted = []

        def transfer_async(self, job_id, transfer_spec):
            self.submitted.append((job_id, transfer_spec))
            return True

        def get_finished(self):
            return [(0, True)]

    worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    worker.worker = TransferWorker()
    worker._job_counter = 0
    worker._transfer_job_counts = {False: 0, True: 0}
    worker._transfer_request_counts = {False: 0, True: 0}
    worker._transfer_block_counts = {False: 0, True: 0}
    worker._jobs = {}
    worker._job_started_s = {}
    worker._job_submit_duration_s = {}
    worker._job_block_counts = {}
    worker._transfer_stats = OffloadingConnectorStats()
    worker._load_job = {}
    worker._store_jobs = {"unused": set()}
    worker._finished_reqs_waiting_for_store = set()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "offloading_connector.time.perf_counter",
        side_effect=[10.0, 10.01, 10.02, 10.25],
    ):
        worker._start_transfer(
            ("request-1", "request-2"),
            [(Spec([1]), Spec([11])), (Spec([2]), Spec([12]))],
            store=False,
        )
        _, finished_recving = worker.get_finished(set())

    assert finished_recving == {"request-1", "request-2"}
    assert len(worker.worker.submitted) == 1
    merged_src, merged_dst = worker.worker.submitted[0][1]
    assert merged_src.block_ids == [1, 2]
    assert merged_dst.block_ids == [11, 12]
    stats = worker.get_kv_connector_stats()
    assert stats.data["h2d_duration_s"] == [0.25]
    assert stats.data["h2d_requests"] == [2]
    assert stats.data["h2d_blocks"] == [2]


def test_adaptive_h2d_defers_only_prefill_sensitive_steps():
    class Spec:
        def __init__(self, block_ids):
            self.block_ids = list(block_ids)

    scheduler = OffloadingConnectorScheduler.__new__(
        OffloadingConnectorScheduler
    )
    scheduler.h2d_scheduling = "adaptive"
    scheduler.h2d_prefill_token_threshold = 2
    scheduler._get_reqs_to_store = lambda output: {}
    scheduler._record_transfer_starts = lambda reqs, store: None

    transfer = {"request-1": (Spec([1]), Spec([11]))}
    decode_output = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"decode-1": 1},
    )
    scheduler._reqs_to_load = transfer
    decode_meta = scheduler.build_connector_meta(decode_output)
    assert decode_meta.h2d_scheduling == "async"

    prefill_output = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"prefill-1": 64},
    )
    scheduler._reqs_to_load = transfer
    prefill_meta = scheduler.build_connector_meta(prefill_output)
    assert prefill_meta.h2d_scheduling == "deferred_sync"


def test_deferred_h2d_submits_after_forward_and_waits_for_completion():
    class Spec:
        def __init__(self, block_ids):
            self.block_ids = list(block_ids)

    class TransferWorker:
        def __init__(self):
            self.submitted = []
            self.waited = []

        def transfer_async(self, job_id, transfer_spec):
            self.submitted.append((job_id, transfer_spec))
            return True

        def wait_for_jobs(self, job_ids):
            self.waited.append(set(job_ids))

        def get_finished(self):
            return [(0, True)]

    worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    worker.worker = TransferWorker()
    worker.batch_h2d_transfers = True
    worker._job_counter = 0
    worker._transfer_job_counts = {False: 0, True: 0}
    worker._transfer_request_counts = {False: 0, True: 0}
    worker._transfer_block_counts = {False: 0, True: 0}
    worker._jobs = {}
    worker._job_started_s = {}
    worker._job_submit_duration_s = {}
    worker._job_block_counts = {}
    worker._transfer_stats = OffloadingConnectorStats()
    worker._load_job = {}
    worker._store_jobs = {}
    worker._finished_reqs_waiting_for_store = set()
    worker._deferred_load = None
    metadata = OffloadingConnectorMetadata(
        reqs_to_load={"request-1": (Spec([1]), Spec([11]))},
        reqs_to_store={},
        h2d_scheduling="deferred_sync",
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "offloading_connector.time.perf_counter",
        side_effect=[10.0, 10.5, 10.6, 10.61, 10.62, 10.7, 10.72, 10.75],
    ):
        worker.start_load_kv(metadata)
        assert worker.worker.submitted == []
        worker.finish_load_kv(metadata)
        assert len(worker.worker.submitted) == 1
        assert worker.worker.waited == [{0}]
        worker.get_finished(set())

    stats = worker.get_kv_connector_stats()
    assert stats.data["h2d_scheduling"] == ["deferred_sync"]
    assert stats.data["h2d_defer_delay_s"] == [0.5]
    assert stats.data["h2d_blocking_wait_s"][0] == pytest.approx(0.02)


def test_offload_worker_batches_both_directions_by_default():
    with patch.dict(os.environ, {"VLLM_CPU_KV_BATCH_TRANSFERS": "1"}):
        os.environ.pop("VLLM_CPU_KV_BATCH_H2D_TRANSFERS", None)
        os.environ.pop("VLLM_CPU_KV_BATCH_D2H_TRANSFERS", None)
        worker = OffloadingConnectorWorker(SimpleNamespace())

    assert worker.batch_h2d_transfers
    assert worker.batch_d2h_transfers


def test_cpu_gpu_block_expansion_preserves_skip_semantics():
    block_ids = np.asarray([0, 1, 3], dtype=np.int64)
    complete = np.empty(12, dtype=np.int64)
    expand_block_ids(block_ids, 4, complete)
    assert complete.tolist() == list(range(8)) + list(range(12, 16))

    skipped = np.empty(10, dtype=np.int64)
    expand_block_ids(block_ids, 4, skipped, skip_count=2)
    assert skipped.tolist() == [2, 3, 4, 5, 6, 7, 12, 13, 14, 15]


def test_async_llm_routes_mnemorag_stats_through_utility_call():
    expected = {"heap_compactions": 3}

    class FakeClient:
        async def call_utility_async(self, method):
            assert method == "get_mnemorag_kv_cache_stats"
            return expected

        def shutdown(self):
            pass

    engine = AsyncLLM.__new__(AsyncLLM)
    engine.engine_core = FakeClient()

    result = asyncio.run(engine.get_mnemorag_kv_cache_stats())
    assert result == expected


def test_async_llm_signals_first_token_from_raw_engine_output():
    async def run():
        event = asyncio.Event()
        processor = OutputProcessor.__new__(OutputProcessor)
        processor.mnemorag_first_token_events = {"request-1": event}
        empty = SimpleNamespace(request_id="request-1", new_token_ids=[])
        token = SimpleNamespace(request_id="request-1", new_token_ids=[7])

        processor._signal_mnemorag_first_token_event(empty)
        assert not event.is_set()
        processor._signal_mnemorag_first_token_event(token)

        assert event.is_set()
        assert isinstance(event.mnemorag_set_time_s, float)
        assert processor.mnemorag_first_token_events == {}

    asyncio.run(run())
