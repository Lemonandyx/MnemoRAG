# SPDX-License-Identifier: Apache-2.0
"""Heat-aware CPU KV offloading manager."""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class HeatOffloadingManager(OffloadingManager):
    """Evict the available CPU block with the lowest heat priority."""

    _TIME_SCALE_S = 1.0
    _ELAPSED_EPSILON_S = 1.0e-12

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend = backend
        self.blocks: dict[BlockHash, BlockStatus] = {}
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self._frequency: dict[BlockHash, float] = {}
        self._last_touch_s: dict[BlockHash, float] = {}
        self.eviction_count = 0

    def _priority(self, block_hash: BlockHash, now_s: float) -> float:
        frequency = max(self._frequency.get(block_hash, 1.0), 1.0)
        elapsed_s = max(
            now_s - self._last_touch_s.get(block_hash, now_s),
            self._ELAPSED_EPSILON_S,
        )
        return frequency / math.log1p(elapsed_s / self._TIME_SCALE_S)

    def _select_victims(self, count: int) -> list[BlockHash] | None:
        now_s = time.monotonic()
        candidates = [
            (self._priority(block_hash, now_s), bytes(block_hash), block_hash)
            for block_hash, block in self.blocks.items()
            if block.ref_cnt == 0
        ]
        if len(candidates) < count:
            return None
        return [entry[2] for entry in heapq.nsmallest(count, candidates)]

    def _remove_metadata(self, block_hash: BlockHash) -> None:
        self._frequency.pop(block_hash, None)
        self._last_touch_s.pop(block_hash, None)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        hashes = list(block_hashes)
        blocks = []
        for block_hash in hashes:
            block = self.blocks[block_hash]
            if not block.is_ready:
                raise RuntimeError(f"CPU KV block {block_hash!r} is not ready")
            block.ref_cnt += 1
            blocks.append(block)
        return self.backend.get_load_store_spec(hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        now_s = time.monotonic()
        for block_hash in dict.fromkeys(block_hashes):
            if block_hash not in self.blocks:
                continue
            self._frequency[block_hash] = self._frequency.get(block_hash, 1.0) + 1.0
            self._last_touch_s[block_hash] = now_s

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            if block.ref_cnt <= 0:
                raise RuntimeError(f"CPU KV block {block_hash!r} is not loaded")
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        hashes = list(dict.fromkeys(block_hashes))
        to_store = [value for value in hashes if value not in self.blocks]
        needed = len(to_store) - self.backend.get_num_free_blocks()
        victims: list[BlockHash] = []
        if needed > 0:
            selected = self._select_victims(needed)
            if selected is None:
                return None
            victims = selected
            for block_hash in victims:
                self.backend.free(self.blocks.pop(block_hash))
                self._remove_metadata(block_hash)
            self.eviction_count += len(victims)

        if victims and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=victims,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(to_store)
        if len(blocks) != len(to_store):
            raise RuntimeError("CPU KV backend returned an invalid allocation")
        now_s = time.monotonic()
        for block_hash, block in zip(to_store, blocks):
            self.blocks[block_hash] = block
            self._frequency[block_hash] = 1.0
            self._last_touch_s[block_hash] = now_s

        return PrepareStoreOutput(
            block_hashes_to_store=to_store,
            store_spec=self.backend.get_load_store_spec(to_store, blocks),
            block_hashes_evicted=victims,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored = []
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or block.is_ready:
                continue
            if success:
                block.ref_cnt = 0
                stored.append(block_hash)
            else:
                self.backend.free(self.blocks.pop(block_hash))
                self._remove_metadata(block_hash)
        if stored and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
