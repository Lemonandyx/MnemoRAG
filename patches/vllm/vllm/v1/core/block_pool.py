# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable, Sequence
import os
import time
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """Map a prefix hash to one block or a dictionary of duplicate blocks.

    Blocks are not deduplicated so allocated block IDs remain stable.
    The single-block representation avoids a dictionary for most hashes.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Return a block for the given hash, or None."""
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """Insert a block into the hash mapping."""
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """Remove the given block ID from the hash mapping."""
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Retain the mapping when block_id does not match.
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Keep any remaining blocks under the same hash.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """Allocate, free, and cache KV blocks.

    Free blocks are kept in eviction order; cached hashes support prefix lookup.
    A group's block size may be a multiple of hash_block_size.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # The placeholder block has ID 0 and must not be freed.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.pattern_metadata_capacity = int(
            os.environ.get(
                "VLLM_KV_PATTERN_METADATA_CAPACITY",
                str(max(65_536, 4 * num_gpu_blocks)),
            )
        )
        if self.pattern_metadata_capacity < 0:
            raise ValueError(
                "VLLM_KV_PATTERN_METADATA_CAPACITY must be non-negative"
            )
        self.pattern_metadata_evictions = {
            "heat": 0,
        }
        self.block_pattern_heat: OrderedDict[
            BlockHashWithGroupId, float
        ] = OrderedDict()
        self.document_heat_accesses = 0
        self.document_heat_blocks = 0
        self.document_heat_above_one_blocks = 0
        self.document_heat_max = 0.0
        self.heat_min_admission_count = int(
            os.environ.get("VLLM_KV_HEAT_MIN_ADMISSION_COUNT", "1")
        )
        if self.heat_min_admission_count < 1:
            raise ValueError(
                "VLLM_KV_HEAT_MIN_ADMISSION_COUNT must be at least 1"
            )

    def _store_pattern_metadata(
        self,
        table: OrderedDict,
        block_hash: BlockHashWithGroupId,
        value: Any,
        category: str,
    ) -> None:
        table.pop(block_hash, None)
        if self.pattern_metadata_capacity == 0:
            return
        table[block_hash] = value
        if len(table) > self.pattern_metadata_capacity:
            table.popitem(last=False)
            self.pattern_metadata_evictions[category] += 1

    def get_mnemorag_stats(self) -> dict[str, Any]:
        queue = self.free_block_queue
        return {
            "eviction_policy": queue.eviction_policy,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_free_blocks": queue.num_free_blocks,
            "heap_entries": len(queue.heap),
            "heap_stale_pops": queue.heap_stale_pops,
            "heap_compactions": queue.heap_compactions,
            "heat_formula": queue.heat_formula,
            "heat_formula_priority_calls": queue.heat_formula_priority_calls,
            "paper_zero_guard_calls": queue.paper_zero_guard_calls,
            "heat_elapsed_epsilon_s": queue.heat_elapsed_epsilon_s,
            "heat_time_scale_s": queue.heat_time_scale_s,
            "pattern_metadata_capacity": self.pattern_metadata_capacity,
            "pattern_heat_entries": len(self.block_pattern_heat),
            "document_heat_accesses": self.document_heat_accesses,
            "document_heat_blocks": self.document_heat_blocks,
            "document_heat_above_one_blocks": (
                self.document_heat_above_one_blocks
            ),
            "document_heat_max": self.document_heat_max,
            "pattern_metadata_evictions": dict(
                self.pattern_metadata_evictions
            ),
        }

    def _get_document_block_heat(
        self, request: Request, block_index: int, block_size: int
    ) -> float | None:
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        spans = extra_args.get("mnemorag_document_heat_spans")
        if not spans:
            return None

        self.document_heat_accesses += 1
        block_start = block_index * block_size
        block_end = block_start + block_size
        weighted_heat = 0.0
        covered_tokens = 0
        for token_start, token_end, heat in spans:
            overlap = max(
                0,
                min(block_end, int(token_end))
                - max(block_start, int(token_start)),
            )
            if overlap:
                weighted_heat += overlap * max(1.0, float(heat))
                covered_tokens += overlap
        if not covered_tokens:
            return None
        self.document_heat_blocks += 1
        document_heat = weighted_heat / covered_tokens
        if document_heat > 1.0:
            self.document_heat_above_one_blocks += 1
        self.document_heat_max = max(self.document_heat_max, document_heat)
        return document_heat

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Return cached blocks for all requested groups, or None on any miss."""
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache newly completed blocks and update their hash metadata."""
        if block_size == self.hash_block_size:
            block_hashes: BlockHashList = request.block_hashes
        else:
            # Combine hashes when this group's blocks span multiple hash blocks.
            assert block_size % self.hash_block_size == 0
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )
        prompt_full_blocks = min(
            num_full_blocks, request.num_prompt_tokens // block_size
        )
        cached_blocks = blocks[:num_cached_blocks]
        for i, blk in enumerate(cached_blocks):
            assert blk.block_hash is not None
            h = blk.block_hash
            if h not in request.block_hashe_added:
                self._record_block_access(
                    h,
                    blk,
                    i,
                    prompt_full_blocks,
                    num_cached_blocks,
                    cache_hit=True,
                    request_position=int(request.request_id),
                    document_heat=self._get_document_block_heat(
                        request, i, block_size
                    ),
                )
                request.block_hashe_added.add(h)
        if num_cached_blocks >= num_full_blocks:
            return

        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            h = blk.block_hash
            if h not in request.block_hashe_added:
                block_index = num_cached_blocks + i
                self._record_block_access(
                    h,
                    blk,
                    block_index,
                    prompt_full_blocks,
                    num_cached_blocks,
                    cache_hit=False,
                    request_position=int(request.request_id),
                    document_heat=self._get_document_block_heat(
                        request, block_index, block_size
                    ),
                )
                request.block_hashe_added.add(h)
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash)
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                )
            )

    def _record_block_access(
        self,
        block_hash: BlockHashWithGroupId,
        block: KVCacheBlock,
        block_index: int,
        prompt_full_blocks: int,
        num_cached_blocks: int,
        cache_hit: bool,
        request_position: int,
        document_heat: float | None = None,
    ) -> None:
        """Update cache metadata for one accessed block."""
        policy = self.free_block_queue.eviction_policy
        if policy == "heat":
            prefix_frequency = self.block_pattern_heat.get(block_hash, 0.0) + 1.0
            heat = prefix_frequency if document_heat is None else document_heat
            self._store_pattern_metadata(
                self.block_pattern_heat,
                block_hash,
                heat,
                "heat",
            )
            block.heat = heat

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Allocate blocks from the free queue without a cache lookup."""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Remove a cached block and reset its hash; return whether it was cached."""
        block_hash = block.block_hash

        if block_hash is None:
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False
        block.reset_hash()
        block.heat = 0.0

        if self.enable_kv_cache_events:
            # KV cache events use one group when hybrid cache management is disabled.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
        """Increase block references and remove newly used blocks from the free queue."""
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # Remove unreferenced blocks from the eviction queue before reuse.
                if block.ref_cnt == 0 and not block.is_null :
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free blocks in eviction order, with the first block evicted first."""
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                released_at = time.monotonic()
                block.last_used_ts = released_at
                block.last_used_monotonic_s = released_at
        freeable_blocks = [
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ]
        if (
            self.free_block_queue.eviction_policy == "heat"
            and self.heat_min_admission_count > 1
        ):
            for block in freeable_blocks:
                if (
                    block.block_hash is not None
                    and block.heat < self.heat_min_admission_count
                ):
                    self._maybe_evict_cached_block(block)
                    # Preserve the hash-level ghost counter, but make this
                    # physical block the first allocation candidate.
                    block.heat = 0.0
        self.free_block_queue.append_n(freeable_blocks)

    def reset_prefix_cache(self) -> bool:
        """Clear prefix-cache hashes if no requests hold blocks."""
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.block_pattern_heat.clear()
        for category in self.pattern_metadata_evictions:
            self.pattern_metadata_evictions[category] = 0

        for block in self.blocks:
            block.reset_hash()
            block.heat = 0.0

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Return the number of free blocks."""
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Return the fraction of KV blocks in use."""

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Return pending events and clear the queue."""
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
