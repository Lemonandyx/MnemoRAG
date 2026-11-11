# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import tempfile
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
            "CPU KV CUDA stream priorities: h2d=%d, d2h=%d",
            self.h2d_stream_priority,
            self.d2h_stream_priority,
        )

        self.sync_h2d = os.environ.get("VLLM_CPU_KV_SYNC_H2D", "0") == "1"
        self.sync_d2h = os.environ.get("VLLM_CPU_KV_SYNC_D2H", "0") == "1"
        self.order_d2h_after_compute = (
            os.environ.get("VLLM_CPU_KV_ORDER_D2H_AFTER_COMPUTE", "1") == "1"
        )
        logger.info("CPU KV D2H waits for compute: %s",
                    self.order_d2h_after_compute)
        if self.sync_h2d or self.sync_d2h:
            logger.info(
                "CPU KV directional synchronization: h2d=%s, d2h=%s",
                self.sync_h2d,
                self.sync_d2h,
            )

        # job_id -> (start, completion) CUDA events. Timing-enabled events let
        # the connector distinguish actual copy time from completion polling.
        self.transfer_events: dict[int, tuple[torch.Event, torch.Event]] = {}
        self.events_pool: list[tuple[torch.Event, torch.Event]] = []
        self.completed_cuda_duration_s: dict[int, float] = {}
        self.staging_refs: dict[int, list[torch.Tensor]] = {}
        self.d2h_postcopies: dict[
            int,
            list[tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]],
        ] = {}
        self.active_staging_job: dict[str, int | None] = {
            "h2d": None,
            "d2h": None,
        }

        self.host_memory_mode = os.environ.get(
            "VLLM_CPU_KV_HOST_MEMORY_MODE", "pinned_eager"
        )
        if self.host_memory_mode not in {
            "pinned_eager",
            "pageable_lazy",
            "hybrid_pinned_lazy",
        }:
            raise ValueError(
                "VLLM_CPU_KV_HOST_MEMORY_MODE must be pinned_eager, "
                "pageable_lazy, or hybrid_pinned_lazy"
            )
        pin_memory = (
            self.host_memory_mode == "pinned_eager"
            and is_pin_memory_available()
        )
        logger.info(
            "CPU KV host memory mode: %s (pin_memory=%s)",
            self.host_memory_mode,
            pin_memory,
        )

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.gpu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        self.cpu_pinned_tensors: list[torch.Tensor] = []
        self.cpu_lazy_tensors: list[torch.Tensor] = []
        self.h2d_staging_tensors: list[torch.Tensor] = []
        self.d2h_staging_tensors: list[torch.Tensor] = []
        self.pinned_tier_blocks = min(
            num_cpu_blocks,
            int(os.environ.get("VLLM_CPU_KV_PINNED_TIER_BLOCKS", "0")),
        )
        self.pinned_tier_subblocks = (
            self.pinned_tier_blocks * self.block_size_factor
        )
        if (
            self.host_memory_mode == "hybrid_pinned_lazy"
            and self.pinned_tier_blocks <= 0
        ):
            raise ValueError(
                "hybrid_pinned_lazy requires a positive pinned tier"
            )
        if self.host_memory_mode == "hybrid_pinned_lazy":
            logger.info(
                "CPU KV hybrid tiers: pinned_blocks=%d, "
                "lazy_blocks=%d",
                self.pinned_tier_blocks,
                num_cpu_blocks - self.pinned_tier_blocks,
            )
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
            if self.host_memory_mode == "hybrid_pinned_lazy":
                pinned_shape = list(cpu_shape)
                lazy_shape = list(cpu_shape)
                pinned_shape[num_blocks_idx] = self.pinned_tier_subblocks
                lazy_shape[num_blocks_idx] -= self.pinned_tier_subblocks
                self.cpu_pinned_tensors.append(
                    torch.zeros(
                        pinned_shape,
                        dtype=gpu_tensor.dtype,
                        device="cpu",
                        pin_memory=is_pin_memory_available(),
                    )
                )
                self.cpu_lazy_tensors.append(
                    self._allocate_lazy_mmap_tensor(
                        lazy_shape,
                        gpu_tensor.dtype,
                        len(self.cpu_lazy_tensors),
                    )
                )
            else:
                factory = (
                    torch.zeros
                    if self.host_memory_mode == "pinned_eager"
                    else torch.empty
                )
                self.cpu_tensors.append(
                    factory(
                        cpu_shape,
                        dtype=gpu_tensor.dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                )

        if self.host_memory_mode != "pinned_eager":
            staging_source = (
                self.cpu_lazy_tensors
                if self.host_memory_mode == "hybrid_pinned_lazy"
                else self.cpu_tensors
            )
            staging_bytes = 0
            staging_max_blocks = int(
                os.environ.get("VLLM_CPU_KV_STAGING_MAX_BLOCKS", "0")
            )
            for gpu_tensor, cpu_tensor, kv_dim in zip(
                self.gpu_tensors,
                staging_source,
                self.kv_dim_before_num_blocks,
            ):
                block_dim = 1 if kv_dim else 0
                staging_blocks = int(gpu_tensor.shape[block_dim])
                if staging_max_blocks > 0:
                    staging_blocks = min(staging_blocks, staging_max_blocks)
                h2d_staging = self._allocate_staging(
                    cpu_tensor, kv_dim, staging_blocks
                )
                d2h_staging = self._allocate_staging(
                    cpu_tensor, kv_dim, staging_blocks
                )
                self.h2d_staging_tensors.append(h2d_staging)
                self.d2h_staging_tensors.append(d2h_staging)
                staging_bytes += (
                    h2d_staging.numel() + d2h_staging.numel()
                ) * h2d_staging.element_size()
            logger.info(
                "CPU KV reusable staging buffers: directions=2, "
                "max_blocks=%d, bytes=%d",
                staging_max_blocks,
                staging_bytes,
            )

    @staticmethod
    def _allocate_lazy_mmap_tensor(
        shape: list[int],
        dtype: torch.dtype,
        tensor_index: int,
    ) -> torch.Tensor:
        """Map the lazy tier without pre-faulting its logical capacity."""
        mmap_dir = os.environ.get("VLLM_CPU_KV_LAZY_MMAP_DIR", "/dev/shm")
        if not os.path.isdir(mmap_dir):
            raise ValueError(
                f"CPU KV lazy mmap directory does not exist: {mmap_dir}"
            )
        numel = math.prod(shape)
        element_size = torch.empty((), dtype=dtype).element_size()
        logical_bytes = numel * element_size
        path = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"mnemorag-cpu-kv-{tensor_index}-",
                dir=mmap_dir,
                delete=False,
            ) as stream:
                path = stream.name
                os.ftruncate(stream.fileno(), logical_bytes)
            tensor = torch.from_file(
                path,
                shared=True,
                size=numel,
                dtype=dtype,
            ).view(shape)
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        backing = "tmpfs_mmap" if mmap_dir == "/dev/shm" else "file_mmap"
        logger.info(
            "CPU KV lazy tier backing: %s, "
            "logical_bytes=%d, directory=%s",
            backing,
            logical_bytes,
            mmap_dir,
        )
        return tensor

    @staticmethod
    def _allocate_staging(
        tensor: torch.Tensor,
        kv_dim_before_num_blocks: bool,
        num_blocks: int,
    ) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[1 if kv_dim_before_num_blocks else 0] = num_blocks
        return torch.empty(
            shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )

    @staticmethod
    def _staging_view(
        staging: torch.Tensor,
        kv_dim_before_num_blocks: bool,
        num_blocks: int,
    ) -> torch.Tensor:
        block_dim = 1 if kv_dim_before_num_blocks else 0
        if num_blocks > staging.shape[block_dim]:
            raise ValueError(
                f"CPU KV staging capacity {staging.shape[block_dim]} is "
                f"smaller than transfer size {num_blocks}"
            )
        return staging.narrow(block_dim, 0, num_blocks)

    def _prepare_staging_reuse(self, direction: str) -> None:
        active_job = self.active_staging_job[direction]
        if active_job is None:
            return
        events = self.transfer_events.get(active_job)
        if events is not None:
            events[1].synchronize()
            if direction == "d2h":
                self._finalize_staging(active_job)
        self.active_staging_job[direction] = None

    @staticmethod
    def _gather_to_staging(
        src: torch.Tensor,
        staging: torch.Tensor,
        block_ids: torch.Tensor,
        kv_dim_before_num_blocks: bool,
    ) -> None:
        if kv_dim_before_num_blocks:
            torch.index_select(src[0], 0, block_ids, out=staging[0])
            torch.index_select(src[1], 0, block_ids, out=staging[1])
        else:
            torch.index_select(src, 0, block_ids, out=staging)

    @staticmethod
    def _scatter_from_staging(
        staging: torch.Tensor,
        dst: torch.Tensor,
        block_ids: torch.Tensor,
        kv_dim_before_num_blocks: bool,
    ) -> None:
        if kv_dim_before_num_blocks:
            dst[0].index_copy_(0, block_ids, staging[0])
            dst[1].index_copy_(0, block_ids, staging[1])
        else:
            dst.index_copy_(0, block_ids, staging)

    def _finalize_staging(self, job_id: int) -> None:
        for staging, dst, kv_dim, block_ids in self.d2h_postcopies.pop(
            job_id, []
        ):
            self._scatter_from_staging(staging, dst, block_ids, kv_dim)
        self.staging_refs.pop(job_id, None)

    @staticmethod
    def _swap_tensor_blocks(
        src: torch.Tensor,
        dst: torch.Tensor,
        mapping: torch.Tensor,
        kv_dim_before_num_blocks: bool,
    ) -> None:
        if kv_dim_before_num_blocks:
            ops.swap_blocks(src[0], dst[0], mapping)
            ops.swap_blocks(src[1], dst[1], mapping)
        else:
            ops.swap_blocks(src, dst, mapping)

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
        use_staging = self.host_memory_mode == "pageable_lazy"
        is_h2d = isinstance(src_spec, CPULoadStoreSpec)
        direction = "h2d" if is_h2d else "d2h"
        if self.host_memory_mode == "hybrid_pinned_lazy":
            cpu_column = 0 if is_h2d else 1
            pinned_mask = src_to_dst[:, cpu_column] < self.pinned_tier_subblocks
            needs_staging = bool((~pinned_mask).any())
        else:
            pinned_mask = None
            needs_staging = use_staging
        if needs_staging:
            self._prepare_staging_reuse(direction)
        staging_refs: list[torch.Tensor] = []
        d2h_postcopies: list[
            tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]
        ] = []

        if self.events_pool:
            start_event, completion_event = self.events_pool.pop()
        else:
            start_event = torch.Event(enable_timing=True)
            completion_event = torch.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if self.host_memory_mode == "hybrid_pinned_lazy":
                assert pinned_mask is not None
                for tensor_idx, (
                    gpu_tensor,
                    pinned_tensor,
                    lazy_tensor,
                    kv_dim,
                ) in enumerate(
                    zip(
                        self.gpu_tensors,
                        self.cpu_pinned_tensors,
                        self.cpu_lazy_tensors,
                        self.kv_dim_before_num_blocks,
                    )
                ):
                    if pinned_mask.any():
                        pinned_mapping = src_to_dst[pinned_mask].copy()
                        if is_h2d:
                            transfer_src, transfer_dst = pinned_tensor, gpu_tensor
                        else:
                            transfer_src, transfer_dst = gpu_tensor, pinned_tensor
                        self._swap_tensor_blocks(
                            transfer_src,
                            transfer_dst,
                            torch.from_numpy(pinned_mapping),
                            kv_dim,
                        )
                    lazy_mask = ~pinned_mask
                    if not lazy_mask.any():
                        continue
                    lazy_mapping = src_to_dst[lazy_mask].copy()
                    lazy_mapping[:, cpu_column] -= self.pinned_tier_subblocks
                    num_transfer_blocks = int(lazy_mapping.shape[0])
                    if is_h2d:
                        staging = self._staging_view(
                            self.h2d_staging_tensors[tensor_idx],
                            kv_dim,
                            num_transfer_blocks,
                        )
                        source_ids = torch.from_numpy(lazy_mapping[:, 0].copy())
                        self._gather_to_staging(
                            lazy_tensor, staging, source_ids, kv_dim
                        )
                        staged_mapping = torch.from_numpy(
                            np.column_stack(
                                (
                                    np.arange(
                                        num_transfer_blocks, dtype=np.int64
                                    ),
                                    lazy_mapping[:, 1],
                                )
                            )
                        )
                        self._swap_tensor_blocks(
                            staging, gpu_tensor, staged_mapping, kv_dim
                        )
                        staging_refs.append(staging)
                    else:
                        staging = self._staging_view(
                            self.d2h_staging_tensors[tensor_idx],
                            kv_dim,
                            num_transfer_blocks,
                        )
                        destination_ids = torch.from_numpy(
                            lazy_mapping[:, 1].copy()
                        )
                        staged_mapping = torch.from_numpy(
                            np.column_stack(
                                (
                                    lazy_mapping[:, 0],
                                    np.arange(
                                        num_transfer_blocks, dtype=np.int64
                                    ),
                                )
                            )
                        )
                        self._swap_tensor_blocks(
                            gpu_tensor, staging, staged_mapping, kv_dim
                        )
                        staging_refs.append(staging)
                        d2h_postcopies.append(
                            (staging, lazy_tensor, kv_dim, destination_ids)
                        )
            else:
                for tensor_idx, (src_tensor, dst_tensor, kv_dim) in enumerate(
                    zip(
                        src_tensors,
                        dst_tensors,
                        self.kv_dim_before_num_blocks,
                    )
                ):
                    transfer_src = src_tensor
                    transfer_dst = dst_tensor
                    transfer_mapping = src_to_dst_tensor
                    if use_staging:
                        num_transfer_blocks = int(src_to_dst.shape[0])
                        if is_h2d:
                            staging = self._staging_view(
                                self.h2d_staging_tensors[tensor_idx],
                                kv_dim,
                                num_transfer_blocks,
                            )
                            source_ids = torch.from_numpy(src_to_dst[:, 0].copy())
                            self._gather_to_staging(
                                src_tensor, staging, source_ids, kv_dim
                            )
                            transfer_src = staging
                            transfer_mapping = torch.from_numpy(
                                np.column_stack(
                                    (
                                        np.arange(
                                            num_transfer_blocks, dtype=np.int64
                                        ),
                                        src_to_dst[:, 1],
                                    )
                                )
                            )
                            staging_refs.append(staging)
                        else:
                            staging = self._staging_view(
                                self.d2h_staging_tensors[tensor_idx],
                                kv_dim,
                                num_transfer_blocks,
                            )
                            destination_ids = torch.from_numpy(
                                src_to_dst[:, 1].copy()
                            )
                            transfer_dst = staging
                            transfer_mapping = torch.from_numpy(
                                np.column_stack(
                                    (
                                        src_to_dst[:, 0],
                                        np.arange(
                                            num_transfer_blocks, dtype=np.int64
                                        ),
                                    )
                                )
                            )
                            staging_refs.append(staging)
                            d2h_postcopies.append(
                                (staging, dst_tensor, kv_dim, destination_ids)
                            )
                    self._swap_tensor_blocks(
                        transfer_src,
                        transfer_dst,
                        transfer_mapping,
                        kv_dim,
                    )
            completion_event.record(stream)

        if staging_refs:
            self.staging_refs[job_id] = staging_refs
        if d2h_postcopies:
            self.d2h_postcopies[job_id] = d2h_postcopies
        if (is_h2d and self.sync_h2d) or (not is_h2d and self.sync_d2h):
            completion_event.synchronize()
            if not is_h2d:
                self._finalize_staging(job_id)

        self.transfer_events[job_id] = (start_event, completion_event)
        if needs_staging:
            self.active_staging_job[direction] = job_id

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, events in self.transfer_events.items():
            start_event, completion_event = events
            if completion_event.query():
                self._finalize_staging(job_id)
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
                if job_id in self.d2h_postcopies:
                    self._finalize_staging(job_id)

    def pop_completed_metrics(self, job_ids: set[int]) -> dict[int, dict[str, float]]:
        return {
            job_id: {
                "cuda_duration_s": self.completed_cuda_duration_s.pop(job_id)
            }
            for job_id in job_ids
            if job_id in self.completed_cuda_duration_s
        }
