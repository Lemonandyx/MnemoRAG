# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention import AttentionBackend
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    if block_ids.size == 0:
        return

    offsets = np.arange(block_size_factor, dtype=output.dtype)
    first_width = block_size_factor - skip_count
    output[:first_width] = (
        int(block_ids[0]) * block_size_factor + offsets[skip_count:]
    )
    if block_ids.size == 1:
        return

    tail = output[first_width:].reshape(-1, block_size_factor)
    np.add(
        block_ids[1:, None] * block_size_factor,
        offsets[None, :],
        out=tail,
        casting="unsafe",
    )


class CpuGpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert cpu_block_size % gpu_block_size == 0
        self.block_size_factor = cpu_block_size // gpu_block_size

        # cuda streams for gpu->cpu and cpu->gpu
        self.d2h_stream_priority = int(
            os.environ.get("VLLM_CPU_KV_D2H_STREAM_PRIORITY", "0")
        )
        self.h2d_stream_priority = int(
            os.environ.get("VLLM_CPU_KV_H2D_STREAM_PRIORITY", "0")
        )
        self.d2h_stream = torch.cuda.Stream(priority=self.d2h_stream_priority)
        self.h2d_stream = torch.cuda.Stream(priority=self.h2d_stream_priority)
        logger.info(
            "[MnemoRAG] CPU KV CUDA stream priorities: h2d=%d, d2h=%d",
            self.h2d_stream_priority,
            self.d2h_stream_priority,
        )

        self.sync_h2d = os.environ.get("VLLM_CPU_KV_SYNC_H2D", "0") == "1"
        self.sync_d2h = os.environ.get("VLLM_CPU_KV_SYNC_D2H", "0") == "1"
        self.order_d2h_after_compute = (
            os.environ.get("VLLM_CPU_KV_ORDER_D2H_AFTER_COMPUTE", "1") == "1"
        )
        logger.info("[MnemoRAG] CPU KV D2H waits for compute: %s",
                    self.order_d2h_after_compute)
        if self.sync_h2d or self.sync_d2h:
            logger.info(
                "[MnemoRAG] CPU KV directional synchronization: h2d=%s, d2h=%s",
                self.sync_h2d,
                self.sync_d2h,
            )

        # job_id -> (start, completion) CUDA events. Timing-enabled events let
        # the connector distinguish actual copy time from completion polling.
        self.transfer_events: dict[int, tuple[torch.Event, torch.Event]] = {}
        self.events_pool: list[tuple[torch.Event, torch.Event]] = []
        self.completed_cuda_duration_s: dict[int, float] = {}

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.gpu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        self.kv_dim_before_num_blocks: list[bool] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            self.gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            elif test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2

                num_blocks_idx = 1
                self.kv_dim_before_num_blocks.append(True)

            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * self.block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            self.cpu_tensors.append(
                torch.zeros(
                    cpu_shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.gpu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream = self.d2h_stream
            src_tensors = self.gpu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor
            if self.order_d2h_after_compute:
                stream.wait_stream(torch.cuda.current_stream())

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * src_block_size_factor
        dst_sub_block_count = dst_blocks.size * dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        if self.events_pool:
            start_event, completion_event = self.events_pool.pop()
        else:
            start_event = torch.Event(enable_timing=True)
            completion_event = torch.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for src_tensor, dst_tensor, kv_dim in zip(
                src_tensors, dst_tensors, self.kv_dim_before_num_blocks
            ):
                if kv_dim:
                    src_key_cache = src_tensor[0]
                    dst_key_cache = dst_tensor[0]
                    ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst_tensor)
                    src_value_cache = src_tensor[1]
                    dst_value_cache = dst_tensor[1]
                    ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst_tensor)
                else:
                    ops.swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor)
            completion_event.record(stream)

        is_h2d = isinstance(src_spec, CPULoadStoreSpec)
        if (is_h2d and self.sync_h2d) or (not is_h2d and self.sync_d2h):
            completion_event.synchronize()

        self.transfer_events[job_id] = (start_event, completion_event)

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, events in self.transfer_events.items():
            start_event, completion_event = events
            if completion_event.query():
                results.append((job_id, True))
                self.completed_cuda_duration_s[job_id] = max(
                    0.0, start_event.elapsed_time(completion_event) / 1000.0
                )
                self.events_pool.append(events)
        for job_id, _ in results:
            del self.transfer_events[job_id]
        return results

    def wait_for_jobs(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            events = self.transfer_events.get(job_id)
            if events is not None:
                events[1].synchronize()

    def pop_completed_metrics(self, job_ids: set[int]) -> dict[int, dict[str, float]]:
        return {
            job_id: {
                "cuda_duration_s": self.completed_cuda_duration_s.pop(job_id)
            }
            for job_id in job_ids
            if job_id in self.completed_cuda_duration_s
        }
