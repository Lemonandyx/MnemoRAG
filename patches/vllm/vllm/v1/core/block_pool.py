# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
import json
import math
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
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
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
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
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
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.fifo_insert_counter = 0

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
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
            "utility": 0,
            "pgdsf": 0,
        }
        self.block_pattern_heat: OrderedDict[
            BlockHashWithGroupId, float
        ] = OrderedDict()
        self.document_heat_accesses = 0
        self.document_heat_blocks = 0
        self.heat_source = os.environ.get(
            "VLLM_KV_HEAT_SOURCE", "document"
        ).strip().lower()
        if self.heat_source not in {"document", "prefix_frequency", "hybrid"}:
            raise ValueError(
                "VLLM_KV_HEAT_SOURCE must be 'document', "
                "'prefix_frequency', or 'hybrid'"
            )
        profile_path = os.environ.get(
            "VLLM_KV_PROFILE_NEXT_USE_PATH",
            os.environ.get("VLLM_KV_ORACLE_NEXT_USE_PATH", ""),
        )
        self.oracle_next_use = {}
        self.oracle_access_count = 0
        self.oracle_match_count = 0
        self.oracle_finite_count = 0
        if profile_path:
            with open(profile_path, encoding="utf-8") as handle:
                self.oracle_next_use = {
                    bytes.fromhex(key): deque(values)
                    for key, values in json.load(handle).items()
                }
            logger.info(
                "[MnemoRAG] Loaded profile-guided next-use data for %d block hashes",
                len(self.oracle_next_use),
            )
        self.block_pattern_utility: OrderedDict[
            BlockHashWithGroupId, tuple[float, float, float]
        ] = OrderedDict()
        # RAGCache PGDSF state: frequency, cumulative miss cost, and samples.
        # KV blocks have a fixed size, so Cost / Size reduces to unit cost.
        self.block_pattern_pgdsf: OrderedDict[
            BlockHashWithGroupId, tuple[float, float, int]
        ] = OrderedDict()
        self.utility_half_life_s = float(
            os.environ.get("VLLM_KV_UTILITY_HALF_LIFE_S", "600")
        )
        self.utility_cost_power = float(
            os.environ.get("VLLM_KV_UTILITY_COST_POWER", "1.0")
        )
        self.utility_output_penalty = float(
            os.environ.get("VLLM_KV_UTILITY_OUTPUT_PENALTY", "0.0")
        )
        self.utility_singleton_penalty = float(
            os.environ.get("VLLM_KV_UTILITY_SINGLETON_PENALTY", "0.0")
        )
        self.heat_min_admission_count = int(
            os.environ.get("VLLM_KV_HEAT_MIN_ADMISSION_COUNT", "1")
        )
        self.heat_tail_probation_blocks = int(
            os.environ.get("VLLM_KV_HEAT_TAIL_PROBATION_BLOCKS", "0")
        )
        if self.utility_half_life_s <= 0:
            raise ValueError("VLLM_KV_UTILITY_HALF_LIFE_S must be positive")
        if self.utility_cost_power < 0:
            raise ValueError("VLLM_KV_UTILITY_COST_POWER must be non-negative")
        if self.utility_output_penalty < 0:
            raise ValueError("VLLM_KV_UTILITY_OUTPUT_PENALTY must be non-negative")
        if self.utility_singleton_penalty < 0:
            raise ValueError(
                "VLLM_KV_UTILITY_SINGLETON_PENALTY must be non-negative"
            )
        if self.heat_min_admission_count < 1:
            raise ValueError(
                "VLLM_KV_HEAT_MIN_ADMISSION_COUNT must be at least 1"
            )
        if self.heat_tail_probation_blocks < 0:
            raise ValueError(
                "VLLM_KV_HEAT_TAIL_PROBATION_BLOCKS must be non-negative"
            )
        self.utility_decay_lambda = math.log(2.0) / self.utility_half_life_s
        self.block_pool_created_time: float

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
            "pattern_metadata_capacity": self.pattern_metadata_capacity,
            "pattern_heat_entries": len(self.block_pattern_heat),
            "heat_source": self.heat_source,
            "document_heat_accesses": self.document_heat_accesses,
            "document_heat_blocks": self.document_heat_blocks,
            "pattern_utility_entries": len(self.block_pattern_utility),
            "pattern_pgdsf_entries": len(self.block_pattern_pgdsf),
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
        return weighted_heat / covered_tokens


    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
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
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
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

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            if self.free_block_queue.eviction_policy == "fifo":
                blk.fifo_insert_seq = self.fifo_insert_counter
                self.fifo_insert_counter += 1
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
            if self.free_block_queue.eviction_policy in {
                "chain_heat", "chain_utility", "chain_pgdsf"
            }:
                block_index = num_cached_blocks + i
                if block_index > 0:
                    parent_hash = make_block_hash_with_group_id(
                        block_hashes[block_index - 1], kv_cache_group_id
                    )
                    parent = self.cached_block_hash_to_block.get_one_block(
                        parent_hash
                    )
                    if parent is None:
                        raise RuntimeError(
                            "Prefix trie parent is missing while caching child"
                        )
                    blk.parent_cache_block = parent
                    parent.cached_child_count += 1
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
        """Update LFU heat, decayed utility, or RAGCache-style PGDSF."""
        policy = self.free_block_queue.eviction_policy
        if self.oracle_next_use:
            self.oracle_access_count += 1
            raw_hash = get_block_hash(block_hash)
            positions = self.oracle_next_use.get(raw_hash)
            if positions is not None:
                self.oracle_match_count += 1
                while positions and positions[0] <= request_position:
                    positions.popleft()
                block.oracle_next_use_request = (
                    float(positions[0]) if positions else float("inf")
                )
                if positions:
                    self.oracle_finite_count += 1
            else:
                block.oracle_next_use_request = float("inf")
            if self.oracle_access_count % 10000 == 0:
                logger.info(
                    "[MnemoRAG] Profile next-use stats: accesses=%d, matches=%d, "
                    "finite=%d, request=%d",
                    self.oracle_access_count,
                    self.oracle_match_count,
                    self.oracle_finite_count,
                    request_position,
                )
        if policy == "chain_pgdsf":
            frequency, total_cost, cost_samples = self.block_pattern_pgdsf.get(
                block_hash, (0.0, 0.0, 0)
            )
            frequency += 1.0

            if not cache_hit:
                cached_span = min(num_cached_blocks, prompt_full_blocks)
                uncached_span = max(
                    1, prompt_full_blocks - num_cached_blocks
                )
                # Analytical stand-in for RAGCache's profiled T(alpha, beta)
                # divided by beta. It reflects attention against the cached
                # prefix plus interactions within newly computed blocks.
                unit_recompute_cost = (
                    1.0 + cached_span + 0.5 * uncached_span
                )
                total_cost += unit_recompute_cost
                cost_samples += 1

            average_cost = (
                total_cost / cost_samples if cost_samples else 1.0
            )
            priority = (
                self.free_block_queue.pgdsf_clock
                + frequency * average_cost
            )
            self._store_pattern_metadata(
                self.block_pattern_pgdsf,
                block_hash,
                (frequency, total_cost, cost_samples),
                "pgdsf",
            )
            block.heat = frequency
            block.recompute_cost = average_cost
            block.is_prompt_block = block_index < prompt_full_blocks
            block.utility_score = priority
            block.utility_version += 1
            return

        if policy in {"heat", "chain_heat"}:
            prefix_frequency = self.block_pattern_heat.get(block_hash, 0.0) + 1.0
            if document_heat is None or self.heat_source == "prefix_frequency":
                heat = prefix_frequency
            elif self.heat_source == "hybrid":
                heat = max(prefix_frequency, document_heat)
            else:
                heat = document_heat
            self._store_pattern_metadata(
                self.block_pattern_heat,
                block_hash,
                heat,
                "heat",
            )
            block.heat = heat
            block.is_tail_probation = (
                self.heat_tail_probation_blocks > 0
                and block_index
                >= max(
                    0,
                    prompt_full_blocks - self.heat_tail_probation_blocks,
                )
            )
            return

        if policy in {"lru", "fifo"}:
            return

        now = time.time() - self.block_pool_created_time
        previous_frequency, previous_access, previous_cost = (
            self.block_pattern_utility.get(block_hash, (0.0, now, 1.0))
        )
        elapsed = max(0.0, now - previous_access)
        frequency = (
            previous_frequency
            * math.exp(-self.utility_decay_lambda * elapsed)
            + 1.0
        )

        # Evicting an early prefix block also makes its cached descendants
        # unusable. This downstream span approximates the prefill work that
        # would have to be recomputed after such an eviction.
        downstream_span = max(1, prompt_full_blocks - block_index)
        recompute_cost = max(previous_cost, float(downstream_span))
        is_prompt_block = block_index < prompt_full_blocks
        log_rank = (
            math.log(frequency)
            + self.utility_decay_lambda * now
            + self.utility_cost_power * math.log(recompute_cost)
            - (0.0 if is_prompt_block else self.utility_output_penalty)
            - (self.utility_singleton_penalty if frequency < 1.5 else 0.0)
        )

        self._store_pattern_metadata(
            self.block_pattern_utility,
            block_hash,
            (frequency, now, recompute_cost),
            "utility",
        )
        block.heat = frequency
        block.recompute_cost = recompute_cost
        block.utility_last_access_ts = now
        block.is_prompt_block = is_prompt_block
        block.utility_score = log_rank
        block.utility_version += 1

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        # In order to only iterate the list once, we duplicated code a bit
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
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash

        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False
        block.reset_hash()
        if self.free_block_queue.eviction_policy in {
            "utility", "chain_heat", "chain_utility", "chain_pgdsf"
        }:
            if block.cached_child_count != 0:
                raise RuntimeError("Evicting a non-leaf prefix trie block")
            block.heat = 0.0
            block.recompute_cost = 1.0
            block.utility_last_access_ts = 0.0
            block.is_prompt_block = True
            block.is_tail_probation = False
            block.utility_score = float("-inf")
            block.utility_version += 1

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null :
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                 block.last_used_ts = time.time() - self.block_pool_created_time
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
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.block_pattern_heat.clear()
        self.block_pattern_utility.clear()
        self.block_pattern_pgdsf.clear()
        for category in self.pattern_metadata_evictions:
            self.pattern_metadata_evictions[category] = 0

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()
            block.parent_cache_block = None
            block.cached_child_count = 0

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
