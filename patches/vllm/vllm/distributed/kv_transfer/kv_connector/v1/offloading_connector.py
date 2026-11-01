# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
import copy
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any, ClassVar

import torch

from vllm.attention import Attention, AttentionBackend, AttentionMetadata
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    h2d_scheduling: str = "async"
    created_at_s: float = 0.0


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    """Worker-observed CPU KV transfer durations for one stats interval."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data = {
            "h2d_duration_s": [],
            "h2d_requests": [],
            "h2d_blocks": [],
            "h2d_submit_duration_s": [],
            "h2d_cuda_duration_s": [],
            "d2h_duration_s": [],
            "d2h_requests": [],
            "d2h_blocks": [],
            "d2h_submit_duration_s": [],
            "d2h_cuda_duration_s": [],
            "h2d_scheduling": [],
            "h2d_scheduling_requests": [],
            "h2d_metadata_delivery_s": [],
            "h2d_defer_delay_s": [],
            "h2d_blocking_wait_s": [],
        }

    def record_transfer(
        self,
        *,
        store: bool,
        duration_s: float,
        request_count: int,
        block_count: int,
        submit_duration_s: float = 0.0,
        cuda_duration_s: float = 0.0,
    ) -> None:
        prefix = "d2h" if store else "h2d"
        self.data[f"{prefix}_duration_s"].append(float(duration_s))
        self.data[f"{prefix}_requests"].append(int(request_count))
        self.data[f"{prefix}_blocks"].append(int(block_count))
        self.data[f"{prefix}_submit_duration_s"].append(
            float(submit_duration_s)
        )
        self.data[f"{prefix}_cuda_duration_s"].append(float(cuda_duration_s))

    def clone_and_reset(self) -> "OffloadingConnectorStats":
        previous = copy.copy(self)
        self.reset()
        return previous

    def record_h2d_scheduling(
        self,
        *,
        mode: str,
        request_count: int,
        metadata_delivery_s: float = 0.0,
        defer_delay_s: float = 0.0,
        blocking_wait_s: float = 0.0,
    ) -> None:
        self.data["h2d_scheduling"].append(mode)
        self.data["h2d_scheduling_requests"].append(int(request_count))
        self.data["h2d_metadata_delivery_s"].append(
            float(metadata_delivery_s)
        )
        self.data["h2d_defer_delay_s"].append(float(defer_delay_s))
        self.data["h2d_blocking_wait_s"].append(float(blocking_wait_s))

    def is_empty(self) -> bool:
        return not (
            self.data["h2d_duration_s"] or self.data["d2h_duration_s"]
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for name, values in other.data.items():
            self.data[name].extend(values)
        return self

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]

    def reduce(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for prefix in ("h2d", "d2h"):
            durations = self.data[f"{prefix}_duration_s"]
            result[f"{prefix}_jobs"] = len(durations)
            result[f"{prefix}_mean_ms"] = (
                1000.0 * sum(durations) / len(durations)
                if durations
                else 0.0
            )
            result[f"{prefix}_p95_ms"] = 1000.0 * self._p95(durations)
            for metric in ("submit_duration_s", "cuda_duration_s"):
                values = self.data[f"{prefix}_{metric}"]
                result[f"{prefix}_{metric.removesuffix('_s')}_mean_ms"] = (
                    1000.0 * sum(values) / len(values) if values else 0.0
                )
        return result


class OffloadingConnector(KVConnectorBase_V1):
    prefer_cross_layer_blocks: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.finish_load_kv(self._connector_metadata)
        self.connector_worker.start_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    def get_mnemorag_transfer_stats(self) -> dict[str, Any] | None:
        if self.connector_scheduler is None:
            return None
        return self.connector_scheduler.get_mnemorag_transfer_stats()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats:
        return OffloadingConnectorStats(data=data or {})


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()
        self._track_gpu_block_reuse = bool(
            getattr(self.manager, "store_activate_on_gpu_pressure", False)
        )
        self._gpu_block_hashes: dict[int, BlockHash] = {}
        self.h2d_scheduling = os.environ.get(
            "VLLM_CPU_KV_H2D_SCHEDULING", "async"
        ).lower()
        if self.h2d_scheduling not in {"async", "sync", "adaptive"}:
            raise ValueError(
                "VLLM_CPU_KV_H2D_SCHEDULING must be async, sync, or adaptive"
            )
        self.h2d_prefill_token_threshold = int(
            os.environ.get(
                "VLLM_CPU_KV_H2D_ADAPTIVE_PREFILL_TOKEN_THRESHOLD", "2"
            )
        )
        if self.h2d_prefill_token_threshold < 1:
            raise ValueError(
                "VLLM_CPU_KV_H2D_ADAPTIVE_PREFILL_TOKEN_THRESHOLD must be positive"
            )

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)
        self._h2d_wait_started_s: dict[ReqId, float] = {}
        self._worker_duration_samples: dict[bool, deque[float]] = {
            False: deque(maxlen=4096),
            True: deque(maxlen=4096),
        }
        self._transfer_stats: dict[bool, dict[str, float | int]] = {
            False: {
                "scheduled_request_operations": 0,
                "scheduled_spec_units": 0,
                "worker_completed_jobs": 0,
                "worker_completed_requests": 0,
                "worker_transferred_blocks": 0,
                "worker_observed_total_s": 0.0,
                "worker_observed_max_s": 0.0,
                "worker_submit_total_s": 0.0,
                "worker_submit_max_s": 0.0,
                "cuda_total_s": 0.0,
                "cuda_max_s": 0.0,
                "scheduler_wait_completed_requests": 0,
                "scheduler_wait_total_s": 0.0,
                "scheduler_wait_max_s": 0.0,
                "async_jobs": 0,
                "sync_jobs": 0,
                "deferred_sync_jobs": 0,
                "scheduling_requests": 0,
                "metadata_delivery_total_s": 0.0,
                "metadata_delivery_max_s": 0.0,
                "defer_delay_total_s": 0.0,
                "defer_delay_max_s": 0.0,
                "blocking_wait_total_s": 0.0,
                "blocking_wait_max_s": 0.0,
            },
            True: {
                "scheduled_request_operations": 0,
                "scheduled_spec_units": 0,
                "worker_completed_jobs": 0,
                "worker_completed_requests": 0,
                "worker_transferred_blocks": 0,
                "worker_observed_total_s": 0.0,
                "worker_observed_max_s": 0.0,
                "worker_submit_total_s": 0.0,
                "worker_submit_max_s": 0.0,
                "cuda_total_s": 0.0,
                "cuda_max_s": 0.0,
            },
        }

    def _record_transfer_starts(
        self, reqs: dict[ReqId, TransferSpec], store: bool
    ) -> None:
        now = time.perf_counter()
        stats = self._transfer_stats[store]
        for req_id, (src_spec, dst_spec) in reqs.items():
            stats["scheduled_request_operations"] += 1
            stats["scheduled_spec_units"] += max(
                len(src_spec.block_ids), len(dst_spec.block_ids)
            )
            if not store:
                self._h2d_wait_started_s.setdefault(req_id, now)

    def _record_h2d_wait_finish(self, req_id: ReqId) -> None:
        started_at = self._h2d_wait_started_s.pop(req_id, None)
        if started_at is None:
            return
        elapsed = max(0.0, time.perf_counter() - started_at)
        stats = self._transfer_stats[False]
        stats["scheduler_wait_completed_requests"] += 1
        stats["scheduler_wait_total_s"] += elapsed
        stats["scheduler_wait_max_s"] = max(
            float(stats["scheduler_wait_max_s"]), elapsed
        )

    def _consume_worker_transfer_stats(
        self, connector_stats: KVConnectorStats | None
    ) -> None:
        if connector_stats is None:
            return
        data = connector_stats.data
        for store, prefix in ((False, "h2d"), (True, "d2h")):
            durations = [float(v) for v in data.get(f"{prefix}_duration_s", ())]
            requests = [int(v) for v in data.get(f"{prefix}_requests", ())]
            blocks = [int(v) for v in data.get(f"{prefix}_blocks", ())]
            submit_durations = [
                float(v)
                for v in data.get(f"{prefix}_submit_duration_s", ())
            ]
            cuda_durations = [
                float(v) for v in data.get(f"{prefix}_cuda_duration_s", ())
            ]
            stats = self._transfer_stats[store]
            stats["worker_completed_jobs"] += len(durations)
            stats["worker_completed_requests"] += sum(requests)
            stats["worker_transferred_blocks"] += sum(blocks)
            stats["worker_observed_total_s"] += sum(durations)
            stats["worker_submit_total_s"] = float(
                stats.get("worker_submit_total_s", 0.0)
            ) + sum(submit_durations)
            stats["cuda_total_s"] = float(
                stats.get("cuda_total_s", 0.0)
            ) + sum(cuda_durations)
            if durations:
                stats["worker_observed_max_s"] = max(
                    float(stats["worker_observed_max_s"]), max(durations)
                )
                self._worker_duration_samples[store].extend(durations)
            if submit_durations:
                stats["worker_submit_max_s"] = max(
                    float(stats.get("worker_submit_max_s", 0.0)),
                    max(submit_durations),
                )
            if cuda_durations:
                stats["cuda_max_s"] = max(
                    float(stats.get("cuda_max_s", 0.0)), max(cuda_durations)
                )
        h2d_stats = self._transfer_stats[False]
        modes = [str(v) for v in data.get("h2d_scheduling", ())]
        request_counts = [
            int(v) for v in data.get("h2d_scheduling_requests", ())
        ]
        defer_delays = [float(v) for v in data.get("h2d_defer_delay_s", ())]
        blocking_waits = [float(v) for v in data.get("h2d_blocking_wait_s", ())]
        metadata_deliveries = [
            float(v) for v in data.get("h2d_metadata_delivery_s", ())
        ]
        for mode in modes:
            key = f"{mode}_jobs"
            h2d_stats[key] = int(h2d_stats.get(key, 0)) + 1
        h2d_stats["scheduling_requests"] = int(
            h2d_stats.get("scheduling_requests", 0)
        ) + sum(request_counts)
        h2d_stats["defer_delay_total_s"] = float(
            h2d_stats.get("defer_delay_total_s", 0.0)
        ) + sum(defer_delays)
        h2d_stats["blocking_wait_total_s"] = float(
            h2d_stats.get("blocking_wait_total_s", 0.0)
        ) + sum(blocking_waits)
        h2d_stats["metadata_delivery_total_s"] = float(
            h2d_stats.get("metadata_delivery_total_s", 0.0)
        ) + sum(metadata_deliveries)
        if defer_delays:
            h2d_stats["defer_delay_max_s"] = max(
                float(h2d_stats.get("defer_delay_max_s", 0.0)),
                max(defer_delays),
            )
        if blocking_waits:
            h2d_stats["blocking_wait_max_s"] = max(
                float(h2d_stats.get("blocking_wait_max_s", 0.0)),
                max(blocking_waits),
            )
        if metadata_deliveries:
            h2d_stats["metadata_delivery_max_s"] = max(
                float(h2d_stats.get("metadata_delivery_max_s", 0.0)),
                max(metadata_deliveries),
            )

    @staticmethod
    def _p95(values: Iterable[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]

    def get_mnemorag_transfer_stats(self) -> dict[str, Any]:
        result = {
            "h2d_scheduling_policy": getattr(self, "h2d_scheduling", "async"),
            "h2d_adaptive_prefill_token_threshold": (
                getattr(self, "h2d_prefill_token_threshold", 2)
            ),
            "measurement": (
                "worker-observed elapsed time from asynchronous transfer "
                "submission until completion polling observes the job; H2D "
                "also reports metadata-to-scheduler completion wait, adaptive "
                "post-forward defer delay, and explicit blocking wait"
            )
        }
        for store, name in ((False, "h2d_reload"), (True, "d2h_offload")):
            stats = dict(self._transfer_stats[store])
            completed = int(stats["worker_completed_jobs"])
            total_s = float(stats["worker_observed_total_s"])
            stats["worker_observed_mean_s"] = (
                total_s / completed if completed else 0.0
            )
            stats["worker_submit_mean_s"] = (
                float(stats.get("worker_submit_total_s", 0.0)) / completed
                if completed
                else 0.0
            )
            stats["cuda_mean_s"] = (
                float(stats.get("cuda_total_s", 0.0)) / completed
                if completed
                else 0.0
            )
            samples = self._worker_duration_samples[store]
            stats["worker_observed_p95_recent_s"] = self._p95(samples)
            stats["worker_observed_recent_samples"] = len(samples)
            if not store:
                wait_completed = int(stats["scheduler_wait_completed_requests"])
                wait_total_s = float(stats["scheduler_wait_total_s"])
                stats["scheduler_wait_mean_s"] = (
                    wait_total_s / wait_completed if wait_completed else 0.0
                )
                stats["scheduler_wait_inflight_requests"] = len(
                    self._h2d_wait_started_s
                )
                deferred_jobs = int(stats.get("deferred_sync_jobs", 0))
                blocking_jobs = deferred_jobs + int(stats.get("sync_jobs", 0))
                scheduling_jobs = sum(
                    int(stats.get(f"{mode}_jobs", 0))
                    for mode in ("async", "sync", "deferred_sync")
                )
                stats["defer_delay_mean_s"] = (
                    float(stats.get("defer_delay_total_s", 0.0))
                    / deferred_jobs
                    if deferred_jobs
                    else 0.0
                )
                stats["blocking_wait_mean_s"] = (
                    float(stats.get("blocking_wait_total_s", 0.0))
                    / blocking_jobs
                    if blocking_jobs
                    else 0.0
                )
                stats["metadata_delivery_mean_s"] = (
                    float(stats.get("metadata_delivery_total_s", 0.0))
                    / scheduling_jobs
                    if scheduling_jobs
                    else 0.0
                )
            result[name] = stats
        return result

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor >= num_blocks
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        self._reqs_being_loaded[request.request_id].update(block_hashes)
        self._next_stored_block_idx[request.request_id] = num_blocks

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            # With async scheduling, some tokens may be missing.
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * self.block_size_factor
            assert len(req.block_hashes) >= num_gpu_blocks

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            if self._track_gpu_block_reuse:
                new_block_hashes = list(new_block_hashes)
                pressure_observed = False
                for idx, block_hash in enumerate(new_block_hashes):
                    offloaded_block_idx = start_block_idx + idx
                    gpu_block_idx = offloaded_block_idx * self.block_size_factor
                    for offset in range(self.block_size_factor):
                        block_id = block_ids[gpu_block_idx + offset]
                        previous_hash = self._gpu_block_hashes.get(block_id)
                        if previous_hash is not None and previous_hash != block_hash:
                            pressure_observed = True
                        self._gpu_block_hashes[block_id] = block_hash
                if pressure_observed:
                    notify = getattr(
                        self.manager, "notify_gpu_cache_pressure", None
                    )
                    if notify is not None:
                        notify()
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            if not store_output.block_hashes_to_store:
                continue
            block_hashes_to_store = set(store_output.block_hashes_to_store)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(src_block_ids)

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        reqs_to_load = self._reqs_to_load
        reqs_to_store = self._get_reqs_to_store(scheduler_output)
        self._record_transfer_starts(reqs_to_load, store=False)
        self._record_transfer_starts(reqs_to_store, store=True)
        has_prefill_work = bool(scheduler_output.scheduled_new_reqs) or any(
            int(num_tokens) >= self.h2d_prefill_token_threshold
            for num_tokens in scheduler_output.num_scheduled_tokens.values()
        )
        h2d_scheduling = self.h2d_scheduling
        if h2d_scheduling == "adaptive":
            h2d_scheduling = "deferred_sync" if has_prefill_work else "async"
        meta = OffloadingConnectorMetadata(
            reqs_to_load=reqs_to_load,
            reqs_to_store=reqs_to_store,
            h2d_scheduling=h2d_scheduling,
            created_at_s=time.perf_counter(),
        )
        self._reqs_to_load = {}
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        self._consume_worker_transfer_stats(
            connector_output.kv_connector_stats
        )
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            self._record_h2d_wait_finish(req_id)
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                )


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()
        # The connector exists only when CPU KV offload is enabled. Batch each
        # scheduler step by default so direct deployments and benchmarks use
        # the same transfer policy.
        batch_all = os.environ.get("VLLM_CPU_KV_BATCH_TRANSFERS", "1") == "1"
        self.batch_h2d_transfers = (
            os.environ.get(
                "VLLM_CPU_KV_BATCH_H2D_TRANSFERS", "1" if batch_all else "0"
            )
            == "1"
        )
        self.batch_d2h_transfers = (
            os.environ.get(
                "VLLM_CPU_KV_BATCH_D2H_TRANSFERS", "1" if batch_all else "0"
            )
            == "1"
        )
        logger.info(
            "[MnemoRAG] CPU KV batch transfers per scheduler step: h2d=%s, d2h=%s",
            self.batch_h2d_transfers,
            self.batch_d2h_transfers,
        )

        self._job_counter = 0
        self._transfer_job_counts = {False: 0, True: 0}
        self._transfer_request_counts = {False: 0, True: 0}
        self._transfer_block_counts = {False: 0, True: 0}

        # job_id -> (request IDs sharing the transfer, store)
        self._jobs: dict[int, tuple[tuple[ReqId, ...], bool]] = {}
        self._job_started_s: dict[int, float] = {}
        self._job_submit_duration_s: dict[int, float] = {}
        self._job_block_counts: dict[int, int] = {}
        self._transfer_stats = OffloadingConnectorStats()
        self._deferred_load: tuple[
            tuple[ReqId, ...], list[TransferSpec], float, float
        ] | None = None
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    @staticmethod
    def _merge_transfer_specs(specs: list[TransferSpec]) -> TransferSpec:
        assert specs
        first_src, first_dst = specs[0]
        assert all(type(src) is type(first_src) for src, _ in specs)
        assert all(type(dst) is type(first_dst) for _, dst in specs)
        src_ids = [
            int(block_id)
            for src, _ in specs
            for block_id in src.block_ids
        ]
        dst_ids = [
            int(block_id)
            for _, dst in specs
            for block_id in dst.block_ids
        ]
        return type(first_src)(src_ids), type(first_dst)(dst_ids)

    def _start_transfer(
        self,
        req_ids: tuple[ReqId, ...],
        specs: list[TransferSpec],
        store: bool,
    ) -> int | None:
        if not req_ids:
            return None
        assert len(req_ids) == len(specs)
        transfer_spec = (
            self._merge_transfer_specs(specs) if len(specs) > 1 else specs[0]
        )
        self._transfer_job_counts[store] += 1
        self._transfer_request_counts[store] += len(req_ids)
        self._transfer_block_counts[store] += len(transfer_spec[0].block_ids)
        job_count = self._transfer_job_counts[store]
        if job_count <= 3 or job_count % 1000 == 0:
            logger.info(
                "[MnemoRAG] CPU KV %s transfer stats: jobs=%d, requests=%d, "
                "blocks=%d, current_group=%d",
                "D2H" if store else "H2D",
                job_count,
                self._transfer_request_counts[store],
                self._transfer_block_counts[store],
                len(req_ids),
            )
        job_id = self._generate_job_id()
        self._jobs[job_id] = (req_ids, store)
        self._job_started_s[job_id] = time.perf_counter()
        self._job_block_counts[job_id] = len(transfer_spec[0].block_ids)
        for req_id in req_ids:
            if store:
                self._store_jobs[req_id].add(job_id)
            else:
                assert req_id not in self._load_job
                self._load_job[req_id] = job_id
        submit_started_s = time.perf_counter()
        assert self.worker.transfer_async(job_id, transfer_spec)
        self._job_submit_duration_s[job_id] = max(
            0.0, time.perf_counter() - submit_started_s
        )
        return job_id

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self._transfer_stats.is_empty():
            return None
        return self._transfer_stats.clone_and_reset()

    def _submit_loads(self, metadata: OffloadingConnectorMetadata) -> list[int]:
        job_ids: list[int] = []
        if self.batch_h2d_transfers:
            req_ids = tuple(metadata.reqs_to_load)
            job_id = self._start_transfer(
                req_ids,
                [metadata.reqs_to_load[req_id] for req_id in req_ids],
                store=False,
            )
            if job_id is not None:
                job_ids.append(job_id)
            return job_ids
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._start_transfer((req_id,), [transfer_spec], store=False)
            if job_id is not None:
                job_ids.append(job_id)
        return job_ids

    def start_load_kv(self, metadata: OffloadingConnectorMetadata):
        assert self._deferred_load is None
        req_ids = tuple(metadata.reqs_to_load)
        if not req_ids:
            return
        metadata_delivery_s = (
            max(0.0, time.perf_counter() - metadata.created_at_s)
            if metadata.created_at_s > 0.0
            else 0.0
        )
        if metadata.h2d_scheduling == "deferred_sync":
            self._deferred_load = (
                req_ids,
                [metadata.reqs_to_load[req_id] for req_id in req_ids],
                time.perf_counter(),
                metadata_delivery_s,
            )
            return

        job_ids = self._submit_loads(metadata)
        blocking_wait_s = 0.0
        if metadata.h2d_scheduling == "sync":
            wait_started_s = time.perf_counter()
            self.worker.wait_for_jobs(set(job_ids))
            blocking_wait_s = max(0.0, time.perf_counter() - wait_started_s)
        self._transfer_stats.record_h2d_scheduling(
            mode=metadata.h2d_scheduling,
            request_count=len(req_ids),
            metadata_delivery_s=metadata_delivery_s,
            blocking_wait_s=blocking_wait_s,
        )

    def finish_load_kv(self, metadata: OffloadingConnectorMetadata):
        if self._deferred_load is None:
            return
        req_ids, specs, deferred_at_s, metadata_delivery_s = self._deferred_load
        self._deferred_load = None
        defer_delay_s = max(0.0, time.perf_counter() - deferred_at_s)
        deferred_meta = OffloadingConnectorMetadata(
            reqs_to_load=dict(zip(req_ids, specs)),
            reqs_to_store={},
            h2d_scheduling="deferred_sync",
        )
        job_ids = self._submit_loads(deferred_meta)
        wait_started_s = time.perf_counter()
        self.worker.wait_for_jobs(set(job_ids))
        blocking_wait_s = max(0.0, time.perf_counter() - wait_started_s)
        self._transfer_stats.record_h2d_scheduling(
            mode="deferred_sync",
            request_count=len(req_ids),
            metadata_delivery_s=metadata_delivery_s,
            defer_delay_s=defer_delay_s,
            blocking_wait_s=blocking_wait_s,
        )

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        if self.batch_d2h_transfers:
            req_ids = tuple(metadata.reqs_to_store)
            self._start_transfer(
                req_ids,
                [metadata.reqs_to_store[req_id] for req_id in req_ids],
                store=True,
            )
            return
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            self._start_transfer((req_id,), [transfer_spec], store=True)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        finished = self.worker.get_finished()
        metric_getter = getattr(self.worker, "pop_completed_metrics", None)
        completed_metrics = (
            metric_getter({job_id for job_id, _ in finished})
            if metric_getter is not None
            else {}
        )
        for job_id, success in finished:
            # we currently do not support job failures
            assert success
            req_ids, store = self._jobs.pop(job_id)
            started_s = self._job_started_s.pop(job_id)
            submit_duration_s = self._job_submit_duration_s.pop(job_id, 0.0)
            block_count = self._job_block_counts.pop(job_id)
            job_metrics = completed_metrics.get(job_id, {})
            self._transfer_stats.record_transfer(
                store=store,
                duration_s=max(0.0, time.perf_counter() - started_s),
                request_count=len(req_ids),
                block_count=block_count,
                submit_duration_s=submit_duration_s,
                cuda_duration_s=float(job_metrics.get("cuda_duration_s", 0.0)),
            )
            for req_id in req_ids:
                if store:
                    req_jobs = self._store_jobs[req_id]
                    req_jobs.remove(job_id)
                    if req_jobs:
                        continue

                    if req_id in self._finished_reqs_waiting_for_store:
                        self._finished_reqs_waiting_for_store.remove(req_id)
                        finished_sending.add(req_id)
                        del self._store_jobs[req_id]
                else:
                    req_job = self._load_job[req_id]
                    assert job_id == req_job
                    del self._load_job[req_id]
                    finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...], bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )
