# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict, deque
from collections.abc import Iterable
import heapq
import json
import math
import os

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


logger = init_logger(__name__)


class LRUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self.min_admission_count = max(
            1, int(os.environ.get("VLLM_CPU_KV_MIN_ADMISSION_COUNT", "1"))
        )
        self.admission_counts: dict[BlockHash, int] | None = (
            {} if self.min_admission_count > 1 else None
        )
        profile_path = os.environ.get("VLLM_CPU_KV_ADMISSION_PROFILE_PATH")
        self.admission_profile: set[BlockHash] | None = None
        self.profile_mode = os.environ.get(
            "VLLM_CPU_KV_PROFILE_MODE", ""
        ).strip().lower()
        if self.profile_mode not in {"", "near", "far"}:
            raise ValueError("VLLM_CPU_KV_PROFILE_MODE must be '', 'near', or 'far'")
        self.profile_positions: dict[BlockHash, deque[int]] | None = None
        self.profile_next_use: dict[BlockHash, float] = {}
        self.profile_versions: dict[BlockHash, int] = {}
        self.profile_heap: list[tuple[float, int, BlockHash]] = []
        self.profile_expiry_heap: list[tuple[float, int, BlockHash]] = []
        horizon = int(os.environ.get("VLLM_CPU_KV_MAX_NEXT_USE_DISTANCE", "0"))
        self.max_next_use_distance = horizon if horizon > 0 else math.inf
        self.min_next_use_distance = max(
            0, int(os.environ.get("VLLM_CPU_KV_MIN_NEXT_USE_DISTANCE", "0"))
        )
        self.min_next_use_activate_after = max(
            0,
            int(
                os.environ.get(
                    "VLLM_CPU_KV_MIN_NEXT_USE_ACTIVATE_AFTER", "0"
                )
            ),
        )
        self.min_next_use_activate_on_full = (
            os.environ.get(
                "VLLM_CPU_KV_MIN_NEXT_USE_ACTIVATE_ON_FULL", "0"
            )
            == "1"
        )
        self.min_load_blocks = max(
            1, int(os.environ.get("VLLM_CPU_KV_MIN_LOAD_BLOCKS", "1"))
        )
        self.store_activate_after_lookups = max(
            0,
            int(
                os.environ.get(
                    "VLLM_CPU_KV_STORE_ACTIVATE_AFTER_LOOKUPS", "0"
                )
            ),
        )
        self.store_activate_on_gpu_pressure = (
            os.environ.get(
                "VLLM_CPU_KV_STORE_ACTIVATE_ON_GPU_PRESSURE", "0"
            )
            == "1"
        )
        self.gpu_cache_pressure_observed = False
        self.gpu_cache_pressure_lookup: int | None = None
        self.log_hits = os.environ.get("VLLM_CPU_KV_LOG_HITS", "0") == "1"
        self.log_after_lookups = max(
            0, int(os.environ.get("VLLM_CPU_KV_LOG_AFTER_LOOKUPS", "0"))
        )
        self.dead_gc_enabled = os.environ.get("VLLM_CPU_KV_DEAD_GC", "1") == "1"
        self.global_dead_gc_enabled = (
            os.environ.get("VLLM_CPU_KV_GLOBAL_DEAD_GC", "0") == "1"
        )
        self.global_dead_gc_on_pressure = (
            os.environ.get("VLLM_CPU_KV_GLOBAL_DEAD_GC_ON_PRESSURE", "0") == "1"
        )
        self.dedup_profile_push = (
            os.environ.get("VLLM_CPU_KV_DEDUP_PROFILE_PUSH", "0") == "1"
        )
        self.fast_profile_advance = (
            os.environ.get("VLLM_CPU_KV_FAST_PROFILE_ADVANCE", "0") == "1"
        )
        if profile_path:
            with open(profile_path, encoding="utf-8") as source:
                raw_profile = json.load(source)
            self.admission_profile = {
                BlockHash(bytes.fromhex(value)) for value in raw_profile
            }
            if self.profile_mode:
                self.profile_positions = {
                    BlockHash(bytes.fromhex(value)): deque(positions)
                    for value, positions in raw_profile.items()
                }
            logger.info(
                "<yx> Loaded CPU KV admission profile with %d block hashes, "
                "mode=%s, band=[%d,%s], min_distance_activate_after=%d, "
                "min_distance_activate_on_full=%s, min_load_blocks=%d, "
                "store_activate_after_lookups=%d, "
                "store_activate_on_gpu_pressure=%s, "
                "global_dead_gc=%s, global_dead_gc_on_pressure=%s, "
                "dedup_profile_push=%s, fast_profile_advance=%s",
                len(self.admission_profile),
                self.profile_mode or "lru",
                self.min_next_use_distance,
                self.max_next_use_distance,
                self.min_next_use_activate_after,
                self.min_next_use_activate_on_full,
                self.min_load_blocks,
                self.store_activate_after_lookups,
                self.store_activate_on_gpu_pressure,
                self.global_dead_gc_enabled,
                self.global_dead_gc_on_pressure,
                self.dedup_profile_push,
                self.fast_profile_advance,
            )
        self.lookup_calls = 0
        self.lookup_blocks = 0
        self.lookup_hit_blocks = 0
        self.lookup_hit_requests = 0
        self.lookup_hit_histogram = {
            "1-3": 0,
            "4-7": 0,
            "8-15": 0,
            "16-31": 0,
            "32+": 0,
        }
        self.store_blocks = 0
        self.warmup_skipped_blocks = 0
        self.rejected_blocks = 0
        self.horizon_rejected_blocks = 0
        self.near_rejected_blocks = 0
        self.dead_blocks_freed = 0
        self.profile_priority_pushes = 0
        self.profile_priority_pushes_deduped = 0
        self.profile_advances_skipped = 0

    def notify_gpu_cache_pressure(self) -> None:
        """Enable CPU stores after the GPU cache starts recycling block IDs."""
        if self.gpu_cache_pressure_observed:
            return
        self.gpu_cache_pressure_observed = True
        self.gpu_cache_pressure_lookup = self.lookup_calls
        logger.info(
            "<yx> CPU KV GPU-pressure activation: lookups=%d, "
            "warmup_skipped=%d",
            self.lookup_calls,
            self.warmup_skipped_blocks,
        )

    def _advance_profile(
        self, block_hashes: Iterable[BlockHash], request_position: int
    ) -> None:
        if self.profile_positions is None:
            return
        for block_hash in block_hashes:
            if self.fast_profile_advance:
                current_next_use = self.profile_next_use.get(block_hash)
                if (
                    current_next_use is not None
                    and current_next_use > request_position
                ):
                    self.profile_advances_skipped += 1
                    continue
            positions = self.profile_positions.get(block_hash)
            if positions is None:
                continue
            while positions and positions[0] <= request_position:
                positions.popleft()
            next_use = float(positions[0]) if positions else math.inf
            resident = block_hash in self.blocks
            previous_next_use = (
                self.profile_next_use.get(block_hash)
                if self.dedup_profile_push and resident
                else None
            )
            self.profile_next_use[block_hash] = next_use
            if resident:
                if self.dedup_profile_push and previous_next_use == next_use:
                    self.profile_priority_pushes_deduped += 1
                else:
                    self._push_profile_priority(block_hash)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        block_hashes = list(block_hashes)
        request_position = self.lookup_calls
        self._advance_profile(block_hashes, request_position)
        if not self.global_dead_gc_on_pressure:
            self._advance_expired_profiles(request_position)
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        self.lookup_calls += 1
        self.lookup_blocks += len(block_hashes)
        served_hit_count = hit_count if hit_count >= self.min_load_blocks else 0
        self.lookup_hit_blocks += served_hit_count
        if hit_count > 0:
            self.lookup_hit_requests += 1
            if hit_count < 4:
                bucket = "1-3"
            elif hit_count < 8:
                bucket = "4-7"
            elif hit_count < 16:
                bucket = "8-15"
            elif hit_count < 32:
                bucket = "16-31"
            else:
                bucket = "32+"
            self.lookup_hit_histogram[bucket] += 1
        if served_hit_count > 0 and self.log_hits:
            logger.info(
                "<yx> CPU KV hit: position=%d, blocks=%d",
                request_position,
                served_hit_count,
            )
        if (
            self.lookup_calls % 1000 == 0
            or self.log_after_lookups > 0
            and self.lookup_calls >= self.log_after_lookups
        ):
            logger.info(
                "<yx> CPU KV stats: lookups=%d, hit_blocks=%d/%d, "
                "resident=%d, stored=%d, warmup_skipped=%d, rejected=%d, "
                "near_rejected=%d, horizon_rejected=%d, dead_freed=%d, "
                "profile_pushes=%d, profile_pushes_deduped=%d, "
                "profile_advances_skipped=%d, "
                "hit_requests=%d, hit_histogram=%s",
                self.lookup_calls,
                self.lookup_hit_blocks,
                self.lookup_blocks,
                len(self.blocks),
                self.store_blocks,
                self.warmup_skipped_blocks,
                self.rejected_blocks,
                self.near_rejected_blocks,
                self.horizon_rejected_blocks,
                self.dead_blocks_freed,
                self.profile_priority_pushes,
                self.profile_priority_pushes_deduped,
                self.profile_advances_skipped,
                self.lookup_hit_requests,
                self.lookup_hit_histogram,
            )
        return served_hit_count

    def _profile_eviction_key(self, block_hash: BlockHash) -> float:
        next_use = self.profile_next_use.get(block_hash, math.inf)
        if math.isinf(next_use):
            return -math.inf
        if self.profile_mode == "far":
            # GPU keeps the nearest future uses. CPU evicts those first and
            # retains the farther finite uses as a complementary tier.
            return next_use
        return -next_use

    def _push_profile_priority(self, block_hash: BlockHash) -> None:
        if self.dedup_profile_push:
            self.profile_priority_pushes += 1
        version = self.profile_versions.get(block_hash, 0) + 1
        self.profile_versions[block_hash] = version
        heapq.heappush(
            self.profile_heap,
            (self._profile_eviction_key(block_hash), version, block_hash),
        )
        next_use = self.profile_next_use.get(block_hash, math.inf)
        if math.isfinite(next_use):
            heapq.heappush(
                self.profile_expiry_heap, (next_use, version, block_hash)
            )

    def _advance_expired_profiles(self, request_position: int) -> None:
        if not self.global_dead_gc_enabled or self.profile_positions is None:
            return
        while (
            self.profile_expiry_heap
            and self.profile_expiry_heap[0][0] <= request_position
        ):
            _, version, block_hash = heapq.heappop(self.profile_expiry_heap)
            if (
                block_hash not in self.blocks
                or self.profile_versions.get(block_hash) != version
            ):
                continue
            self._advance_profile((block_hash,), request_position)
            self._free_dead_blocks((block_hash,))

    def _select_profile_victims(self, count: int) -> list[BlockHash] | None:
        victims: list[BlockHash] = []
        busy_entries: list[tuple[float, int, BlockHash]] = []
        while self.profile_heap and len(victims) < count:
            entry = heapq.heappop(self.profile_heap)
            _, version, block_hash = entry
            block = self.blocks.get(block_hash)
            if block is None or self.profile_versions.get(block_hash) != version:
                continue
            if block.ref_cnt != 0:
                busy_entries.append(entry)
                continue
            victims.append(block_hash)
        for entry in busy_entries:
            heapq.heappush(self.profile_heap, entry)
        return victims if len(victims) == count else None

    def _free_dead_blocks(
        self, block_hashes: Iterable[BlockHash]
    ) -> list[BlockHash]:
        if not self.dead_gc_enabled or self.profile_positions is None:
            return []

        freed: list[BlockHash] = []
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if (
                block is not None
                and block.is_ready
                and block.ref_cnt == 0
                and math.isinf(
                    self.profile_next_use.get(block_hash, math.inf)
                )
            ):
                self.backend.free(self.blocks.pop(block_hash))
                self.profile_versions[block_hash] = (
                    self.profile_versions.get(block_hash, 0) + 1
                )
                freed.append(block_hash)

        self.dead_blocks_freed += len(freed)
        if freed and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=freed,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )
        return freed

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        block_hashes = list(block_hashes)
        if self.admission_counts is not None:
            for block_hash in block_hashes:
                count = self.admission_counts.get(block_hash, 0)
                if count < self.min_admission_count:
                    self.admission_counts[block_hash] = count + 1
        for block_hash in reversed(block_hashes):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        block_hashes = list(block_hashes)
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1
        self._free_dead_blocks(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes = list(block_hashes)
        current_position = max(0, self.lookup_calls - 1)
        if (
            self.lookup_calls < self.store_activate_after_lookups
            or self.store_activate_on_gpu_pressure
            and not self.gpu_cache_pressure_observed
        ):
            self.warmup_skipped_blocks += len(block_hashes)
            blocks = self.backend.allocate_blocks([])
            store_spec = self.backend.get_load_store_spec([], blocks)
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=store_spec,
                block_hashes_evicted=[],
            )
        if self.min_next_use_activate_on_full:
            min_distance_active = self.backend.get_num_free_blocks() == 0
        else:
            min_distance_active = (
                current_position >= self.min_next_use_activate_after
            )
        active_min_next_use_distance = (
            self.min_next_use_distance if min_distance_active else 0
        )
        self._advance_profile(block_hashes, current_position)
        if (
            self.global_dead_gc_enabled
            and self.global_dead_gc_on_pressure
            and self.backend.get_num_free_blocks() < len(block_hashes)
        ):
            self._advance_expired_profiles(current_position)
        self._free_dead_blocks(block_hashes)
        if self.admission_profile is not None:
            self.rejected_blocks += sum(
                block_hash not in self.admission_profile
                for block_hash in block_hashes
            )
        if self.profile_positions is not None and not math.isinf(
            self.max_next_use_distance
        ):
            self.horizon_rejected_blocks += sum(
                math.isfinite(self.profile_next_use.get(block_hash, math.inf))
                and self.profile_next_use.get(block_hash, math.inf)
                - current_position
                > self.max_next_use_distance
                for block_hash in block_hashes
            )
        if self.profile_positions is not None and active_min_next_use_distance > 0:
            self.near_rejected_blocks += sum(
                0
                <= self.profile_next_use.get(block_hash, math.inf)
                - current_position
                < active_min_next_use_distance
                for block_hash in block_hashes
            )
        # filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes
            if block_hash not in self.blocks
            and (
                self.admission_profile is None
                or block_hash in self.admission_profile
            )
            and (
                self.admission_counts is None
                or self.admission_counts.get(block_hash, 0)
                >= self.min_admission_count
            )
            and (
                self.profile_positions is None
                or math.isfinite(
                    self.profile_next_use.get(block_hash, math.inf)
                )
            )
            and (
                self.profile_positions is None
                or self.profile_next_use.get(block_hash, math.inf)
                - current_position
                >= active_min_next_use_distance
            )
            and (
                self.profile_positions is None
                or self.profile_next_use.get(block_hash, math.inf)
                - current_position
                <= self.max_next_use_distance
            )
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # build list of blocks to evict
        to_evict = []
        if num_blocks_to_evict > 0:
            if self.profile_positions is not None:
                selected = self._select_profile_victims(num_blocks_to_evict)
                if selected is None:
                    return None
                to_evict = selected
            else:
                for block_hash, block in self.blocks.items():
                    if block.ref_cnt == 0:
                        to_evict.append(block_hash)
                        num_blocks_to_evict -= 1
                        if num_blocks_to_evict == 0:
                            break
                else:
                    # we could not evict enough blocks
                    return None

            if len(to_evict) == 0:
                return None

        # evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block
            if self.profile_positions is not None:
                self._push_profile_priority(block_hash)
        self.store_blocks += len(block_hashes_to_store)

        # build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )
        self._free_dead_blocks(stored_block_hashes)

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
