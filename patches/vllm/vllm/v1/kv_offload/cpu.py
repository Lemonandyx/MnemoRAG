# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.attention import AttentionBackend
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.heat_manager import HeatOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
        if not num_cpu_blocks:
            raise ValueError(
                "num_cpu_blocks must be specified in kv_connector_extra_config"
            )
        self.num_cpu_blocks: int = int(num_cpu_blocks)
        self._manager: OffloadingManager | None = None
        self._handler: OffloadingHandler | None = None
        self.eviction_policy = str(
            self.extra_config.get("eviction_policy", "lru")
        ).lower()

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
            )
            managers = {
                "lru": LRUOffloadingManager,
                "arc": ARCOffloadingManager,
                "heat": HeatOffloadingManager,
            }
            manager_type = managers.get(self.eviction_policy)
            if manager_type is None:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    "Supported policies: lru, arc, heat"
                )
            self._manager = manager_type(
                backend=backend,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "CPU offloading is supported only on CUDA-compatible GPUs"
                )
            self._handler = CpuGpuOffloadingHandler(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
