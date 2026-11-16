# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache utilities."""

import copy
import heapq
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
import time
from typing import Any, NewType, TypeAlias, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# A distinct type for prefix-cache block hashes.
BlockHash = NewType("BlockHash", bytes)

# Block hash combined with its cache-group ID.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# Accept integer hashes for compatibility with older cache keys.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Append a four-byte group ID to a block hash."""
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# First-block hash seed, shared across processes when PYTHONHASHSEED is set.
NONE_HASH: BlockHash


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn is sha256_cbor:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using sha256_cbor as the hash function."
            "Consider setting PYTHONHASHSEED to a fixed value for "
            "reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    in_free_queue: bool = False
    last_used_ts: float = 0.0
    # Absolute monotonic time of the most recent release.
    last_used_monotonic_s: float = 0.0
    heat: float = 0.0

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """Doubly linked free-block queue with LRU and heat-based eviction.

    Blocks carry their own links, avoiding separate queue nodes.
    LRU follows insertion order; heat-based eviction uses access counts and recency.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        # Select LRU or heat-based eviction for this run.
        self.eviction_policy = os.environ.get(
            "VLLM_KV_FREE_BLOCK_POLICY", "heat"
        ).strip().lower()
        if self.eviction_policy not in {"heat", "lru"}:
            raise ValueError(
                "VLLM_KV_FREE_BLOCK_POLICY must be 'heat' or 'lru'"
            )
        self.heat_formula = "paper_log_elapsed"
        self.heat_elapsed_epsilon_s = float(
            os.environ.get(
                "VLLM_KV_HEAT_ELAPSED_EPSILON_S",
                "1e-12",
            )
        )
        self.heat_time_scale_s = float(
            os.environ.get(
                "VLLM_KV_HEAT_TIME_SCALE_S",
                "1.0",
            )
        )
        if self.heat_elapsed_epsilon_s <= 0:
            raise ValueError("VLLM_KV_HEAT_ELAPSED_EPSILON_S must be positive")
        if self.heat_time_scale_s <= 0:
            raise ValueError("VLLM_KV_HEAT_TIME_SCALE_S must be positive")
        self.heat_formula_priority_calls = 0
        self.paper_zero_guard_calls = 0
        self._clock = time.monotonic
        self.heap_compaction_factor = int(
            os.environ.get("VLLM_KV_HEAP_COMPACTION_FACTOR", "4")
        )
        self.heap_compaction_min_stale = int(
            os.environ.get("VLLM_KV_HEAP_COMPACTION_MIN_STALE", "1024")
        )
        if self.heap_compaction_factor < 2:
            raise ValueError("VLLM_KV_HEAP_COMPACTION_FACTOR must be at least 2")
        if self.heap_compaction_min_stale < 0:
            raise ValueError(
                "VLLM_KV_HEAP_COMPACTION_MIN_STALE must be non-negative"
            )
        self._all_blocks = blocks
        self.heap: list[tuple[float, float | int, int, KVCacheBlock]] = []
        self.heap_compactions = 0
        self.heap_stale_pops = 0
        for i in range(self.num_free_blocks):
            blocks[i].in_free_queue = True
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]
            blocks[i].last_used_ts = 0.0
            blocks[i].last_used_monotonic_s = self._clock()
            if self.eviction_policy == "heat":
                heapq.heappush(
                    self.heap,
                    (self.calc_priority(blocks[i]), blocks[i].heat,
                     blocks[i].block_id, blocks[i]),
                )

        # Sentinel nodes keep real blocks linked to a predecessor and successor.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def _heat_formula_priority(
        self, block: KVCacheBlock, now_s: float | None = None
    ) -> float:
        """Compute a block's heat-and-recency priority."""
        self.heat_formula_priority_calls += 1
        if block.heat <= 0:
            return float("-inf")
        current = self._clock() if now_s is None else float(now_s)
        elapsed = max(
            current - float(block.last_used_monotonic_s),
            self.heat_elapsed_epsilon_s,
        )
        if elapsed <= self.heat_elapsed_epsilon_s:
            self.paper_zero_guard_calls += 1
        return block.heat / math.log1p(elapsed / self.heat_time_scale_s)

    def calc_priority(
        self, block: KVCacheBlock, now_s: float | None = None
    ) -> float:
        return self._heat_formula_priority(block, now_s)

    def _uses_dynamic_paper_priority(self) -> bool:
        return self.eviction_policy == "heat"

    def _refresh_dynamic_paper_heap(self) -> None:
        """Refresh eviction priorities using a common timestamp."""
        if not self._uses_dynamic_paper_priority():
            return
        now_s = self._clock()
        self.heap = [
            (
                self.calc_priority(block, now_s),
                self._heap_version(block),
                block.block_id,
                block,
            )
            for block in self._all_blocks
            if self._is_heap_candidate(block)
        ]
        heapq.heapify(self.heap)

    def _heap_version(self, block: KVCacheBlock) -> float:
        return block.heat

    def _is_heap_candidate(self, block: KVCacheBlock) -> bool:
        return block.in_free_queue

    def _maybe_compact_heap(self) -> None:
        if self.eviction_policy == "lru":
            return
        active_limit = self.heap_compaction_factor * max(
            1, self.num_free_blocks
        )
        stale_limit = self.num_free_blocks + self.heap_compaction_min_stale
        if len(self.heap) <= max(active_limit, stale_limit):
            return

        self.heap = [
            (
                self.calc_priority(block),
                self._heap_version(block),
                block.block_id,
                block,
            )
            for block in self._all_blocks
            if self._is_heap_candidate(block)
        ]
        heapq.heapify(self.heap)
        self.heap_compactions += 1

    def popleft(self, *, refresh_dynamic_priority: bool = True) -> KVCacheBlock:
        if self.eviction_policy == "lru":
            first_block = self.fake_free_list_head.next_free_block
            if (
                first_block is self.fake_free_list_tail
                or first_block is None
            ):
                assert self.num_free_blocks == 0, (
                    f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                    "with the free list."
                )
                raise ValueError("No free blocks available")
            self.remove(first_block)
            return first_block

        if refresh_dynamic_priority:
            self._refresh_dynamic_paper_heap()

        while True:
            if not self.heap:
                raise ValueError("No free blocks available")

            _, heap_version, _, block = heapq.heappop(self.heap)
            # Entries are invalidated lazily when the block leaves the free
            # queue or its heat changes.
            if not block.in_free_queue:
                self.heap_stale_pops += 1
                continue
            if heap_version != block.heat:
                self.heap_stale_pops += 1
                continue
            if block.prev_free_block is None or block.next_free_block is None:
                block.in_free_queue = False
                continue
            self.remove(block)

            return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []

        if self.num_free_blocks < n:
            raise ValueError("Not enough free blocks available")

        ret: list[KVCacheBlock] = []

        self._refresh_dynamic_paper_heap()
        for _ in range(n):
            block = self.popleft(refresh_dynamic_priority=False)
            ret.append(block)

        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block from the free queue."""
        if block.prev_free_block is None or block.next_free_block is None:
            # Only blocks already in the free queue can be removed.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block

        block.prev_free_block = block.next_free_block = None
        block.in_free_queue = False
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        assert not block.in_free_queue
        assert block.prev_free_block is None
        assert block.next_free_block is None

        block.in_free_queue = True

        if self.eviction_policy == "heat":
            heapq.heappush(
                self.heap,
                (self.calc_priority(block), block.heat, block.block_id, block),
            )

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None

        last_block.next_free_block = block
        block.prev_free_block = last_block

        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1
        self._maybe_compact_heap()

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        if not blocks:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )

        for block in blocks:
            assert not block.in_free_queue
            assert block.prev_free_block is None
            assert block.next_free_block is None

            block.in_free_queue = True

            if self.eviction_policy == "heat":
                heapq.heappush(
                    self.heap,
                    (self.calc_priority(block), block.heat, block.block_id, block),
                )

            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)
        self._maybe_compact_heap()

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return free blocks in queue order."""
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the request needs extra block-hash keys."""

    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Return multimodal block keys and the next input index."""
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Skip multimodal keys when the block starts after the final input.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length < start_token_idx:
        return extra_keys, start_mm_idx

    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                curr_mm_idx += 1
                continue

            extra_keys.append(mm_feature.identifier)

            if end_token_idx >= offset + length:
                # Advance when the block also contains the end of this input.
                curr_mm_idx += 1
            else:
                break
        else:
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Return the LoRA name as a hash key, or an empty list."""
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Return prompt-embedding hash keys for the specified token range."""
    if request.prompt_embeds is None:
        return []
    block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
    embeds_bytes = tensor_data(block_prompt_embeds).tobytes()
    return [embeds_bytes]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Return request-specific block-hash keys and the next multimodal index."""
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Hash the parent prefix, current tokens, and optional extra keys."""
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """Create a function that hashes newly completed request blocks."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Generated tokens can overlap only the final multimodal input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                break

            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """Return the maximum KV memory requirement in bytes."""
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """Estimate the longest request that fits in the available KV memory."""

    def fits_in_memory(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    current_max = vllm_config.model_config.max_model_len
    left, right = 1, current_max

    if not fits_in_memory(left):
        return 0

    result = 1
    while left <= right:
        mid = (left + right) // 2
        if fits_in_memory(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """Raise ValueError if KV memory cannot hold a maximum-length request."""

    if not kv_cache_spec:
        return

    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when "
            "initializing the engine."
        )

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(
            vllm_config, kv_cache_spec, available_memory
        )
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}."
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory / GiB_bytes:.2f} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory / GiB_bytes:.2f} GiB). "
            f"{estimated_msg} "
            f"Try increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine."
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """Create a KV cache specification for each group of matching layers."""
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Check whether layers share a KV cache type.

    Full attention with and without a sliding window is treated as one type.
    """

    if not kv_cache_spec:
        # Models without KV cache are treated as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Return the maximum concurrency supported by the KV cache configuration."""
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """Apply num_gpu_blocks_override when configured."""
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """Return the number of KV blocks that fit in the available memory."""
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """Return the common KV cache page size."""
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group layers with identical KV cache specifications."""

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """Group layers with one KV cache type and different hidden sizes."""

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Check whether all layers have the same KV cache page size."""

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Increase block sizes to equalize KV cache page sizes.

    Raise NotImplementedError if the specifications cannot be unified.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group attention layers into KV cache groups with equal page sizes.

    Layers in each group share an attention type and block size. Groups use
    the same memory per block and the same number of layers; incomplete groups
    are padded. Cache-hit lookup supports one attention type or full attention
    paired with one other type.
    """
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split attention groups into equal-sized cache groups, padding as needed.
    # The initial group size is the smallest attention group.
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.25:
        # Prefer fewer padding layers when group sizes are close.
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # Interleave layers across groups to reduce per-stage padding.
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Build the KV cache configuration from layer groups and available memory."""
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Layers of one cache type may need different memory for different hidden sizes.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # Share each memory pool across one layer from each cache group.
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """Convert sliding-window specifications to full attention for hybrid models."""

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """Group model layers by compatible KV cache specifications."""
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # Combine layers that require the same number of token slots.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # The cache manager requires a common page size.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # Use equally sized layer groups for different attention types.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """Build the scheduler's KV cache configuration."""
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # Worker configurations differ only in layer names.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # Use any representative layer of this uniform cache type.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Log the resolved KV cache configuration."""
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info_once("GPU KV cache size: %s tokens", num_tokens_str, scope="local")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Build per-worker KV cache configurations with a shared block count.

    Group the combined layer specifications, then size each worker's cache
    using the smallest block count that fits across workers.
    """

    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
        kv_cache_specs, available_memory
    ):
        check_enough_kv_cache_memory(
            vllm_config, kv_cache_spec_one_worker, available_memory_one_worker
        )

    # Merge layer specifications across workers and parallel stages.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    kv_cache_configs: list[KVCacheConfig] = []
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
        kv_cache_specs, available_memory
    ):
        kv_cache_groups_one_worker: list[KVCacheGroupSpec] = []
        for group in global_kv_cache_groups:
            group_layer_names_one_worker = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in kv_cache_spec_one_worker
            ]
            kv_cache_groups_one_worker.append(
                KVCacheGroupSpec(group_layer_names_one_worker, group.kv_cache_spec)
            )
        assert sum(
            len(group.layer_names) for group in kv_cache_groups_one_worker
        ) == len(kv_cache_spec_one_worker), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups_one_worker, available_memory_one_worker
            )
        )

    # Use the smallest worker block count and shrink tensors proportionally.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """Lazily combine consecutive block hashes for a larger block size.

    The target block size must be an integer multiple of the hash block size.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        base = idx * self.scale_factor
        end = base + self.scale_factor
        merged_hash: bytes = self.block_hashes[base]
        for i in range(base + 1, end):
            merged_hash += self.block_hashes[i]
        return BlockHash(merged_hash)


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
